import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import fitz  # PyMuPDF

# STEP 1: Convert ECG PDF to high-res image using PyMuPDF
def pdf_to_image_fitz(pdf_path, output_image_path, zoom=4):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_image_path)
    return output_image_path

# STEP 2: Plot and save grid overlay for visual inspection
def plot_grid_overlay(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(15, 18))
    ax.imshow(img_rgb)

    # Horizontal lines (Y-axis every 20px from 600 to 800)
    for y in range(600, 3081, 20):
        ax.axhline(y, color='orange', linestyle='--', linewidth=0.8)
        ax.text(110, y, f'{y}', color='orange', fontsize=7, verticalalignment='bottom')

    # Vertical lines (X-axis every 50px from 0 to 100)
    for x in range(0, 101, 50):
        ax.axvline(x, color='blue', linestyle='--', linewidth=0.8)
        ax.text(x, 590, f'{x}', color='blue', fontsize=9, rotation=90)

    ax.set_title("Grid Overlay: Horizontal Lines (Y=600–800), Vertical Lines (X=0–100)")
    grid_path = os.path.join(os.path.dirname(image_path), "grid_overlay_preview.png")
    plt.savefig(grid_path)
    plt.close()
    print(f"Grid overlay saved to: {grid_path}")
# STEP 3: Crop ECG leads using precise vertical ranges (no fixed division)
def crop_12_leads_precise(image_path, output_folder):
    img = cv2.imread(image_path)
    os.makedirs(output_folder, exist_ok=True)
    img_height, img_width = img.shape[:2]

    # Manually determined Y ranges for each lead (clean signal zones)
    y_ranges = [
        (660, 780), (840, 988), (990, 1120),   # I, II, III
        (1180, 1310), (1380, 1440), (1510, 1610), # aVR, aVL, aVF
        (1670, 1820), (1890, 2080), (2150,2350), # V1, V2, V3
        (2415, 2625), (2690, 2860), (2920, 3060) # V4, V5, V6
    ]
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    lead_paths = []
    for (y1, y2), lead_name in zip(y_ranges, lead_names):
        lead_img = img[y1:y2, 93:]  # full width
        path = os.path.join(output_folder, f"lead_{lead_name}.png")
        cv2.imwrite(path, lead_img)
        lead_paths.append((lead_name, path))

    return lead_paths

# STEP 4 (revised): Estimate pixels-per-mm from the grid pattern (no amplitude assumption)
def estimate_pixel_scale_from_grid(image_path, min_mm_px=4, max_mm_px=80, debug=False):
    """
    Returns: pixels_per_mm_x, pixels_per_mm_y
    - Works by detecting the colored grid and measuring its fundamental spacing via autocorrelation.
    - min_mm_px / max_mm_px bound the expected 1mm box size in pixels (tune if needed).
    """
    import numpy as np
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # --- 1) Emphasize the red/pink grid (common on ECG paper). Fallback to intensity if needed. ---
    b, g, r = cv2.split(img)
    # Heuristic mask: grid is reddish (R higher than G/B)
    red_dom = (r.astype(np.int16) > g.astype(np.int16) + 20) & (r.astype(np.int16) > b.astype(np.int16) + 20)
    grid_mask = np.zeros(r.shape, dtype=np.uint8)
    grid_mask[red_dom] = 255

    # If mask is too sparse, try a softer condition
    if grid_mask.mean() < 2:  # <2% white: maybe grid isn't strongly red
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # pick thin line-like structures by enhancing high frequencies
        highpass = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        highpass = cv2.convertScaleAbs(highpass)
        _, grid_mask = cv2.threshold(highpass, 0, 255, cv2.THRESH_OTSU)

    # Clean up: thin noise out, keep lines
    grid_mask = cv2.medianBlur(grid_mask, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 2) Project to 1-D profiles ---
    # Sum of white pixels per column/row -> peaks at grid lines
    proj_x = grid_mask.sum(axis=0).astype(np.float32)  # width-long, for vertical grid lines (spacing along X)
    proj_y = grid_mask.sum(axis=1).astype(np.float32)  # height-long, for horizontal grid lines (spacing along Y)

    def fundamental_period_from_projection(proj, min_px, max_px):
        # Detrend
        proj = proj - np.median(proj)
        # Autocorrelation (positive lags)
        ac = np.correlate(proj, proj, mode='full')
        ac = ac[ac.size // 2 + 1:]

        # Bound search window for the 1mm spacing
        lo = int(max(min_px, 1))
        hi = int(min(max_px, ac.size - 1))
        if hi <= lo:
            # Fallback: guess from global max in a reasonable range
            lo, hi = 3, min(200, ac.size - 1)

        lag_main = np.argmax(ac[lo:hi]) + lo

        # Many times the strongest peak is the bold 5mm line spacing.
        # If so, check ~lag_main/5 as the true 1mm spacing.
        candidate = max(3, int(round(lag_main / 5)))
        if candidate < lag_main:
            # Compare autocorr energy
            if ac[candidate] > 0.4 * ac[lag_main]:
                return float(candidate)

        return float(lag_main)

    px_per_mm_x = fundamental_period_from_projection(proj_x, min_mm_px, max_mm_px)
    px_per_mm_y = fundamental_period_from_projection(proj_y, min_mm_px, max_mm_px)

    if debug:
        print(f"[Grid scale] px/mm X: {px_per_mm_x:.3f}, Y: {px_per_mm_y:.3f}")

    return px_per_mm_x, px_per_mm_y

# STEP 5 (revised): Extract ECG waveform using grid-based scale (no 2 mV assumption)
def extract_signal(
    image_path,
    pixels_per_mm_x=None,
    pixels_per_mm_y=None,
    mm_per_mv=10,
    mm_per_sec=25,
    interpolate_gaps=True
):
    """
    Returns: time_sec (1D numpy array), mv (1D numpy array)
    - If pixels_per_mm_x / pixels_per_mm_y are None, they are measured from the grid in this image.
    - Suppresses reddish grid before thresholding.
    - Uses intensity-weighted centroid per column for subpixel y.
    - Optionally interpolates over columns where the trace wasn't detected.
    """

    # 0) Measure px/mm from this image if not provided
    if pixels_per_mm_x is None or pixels_per_mm_y is None:
        pixels_per_mm_x, pixels_per_mm_y = estimate_pixel_scale_from_grid(
            image_path, debug=False
        )

    # 1) Read and suppress red/pink grid
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    b, g, r = cv2.split(img)
    # Heuristic: grid tends to be reddish; wash it out to white
    red_dom = (r.astype(np.int16) > g.astype(np.int16) + 20) & (r.astype(np.int16) > b.astype(np.int16) + 20)
    img_clean = img.copy()
    img_clean[red_dom] = [255, 255, 255]

    # 2) Grayscale, invert (trace becomes bright), threshold
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray

    # Adaptive threshold is more robust than a fixed value across scans
    binary = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 5
    )

    # 3) (Optional) light morphology to clean specks
    binary = cv2.medianBlur(binary, 3)

    # 4) Column-wise centerline: intensity-weighted centroid (subpixel), fallback to median of white pixels
    h, w = binary.shape
    inv_float = inv.astype(np.float32) / 255.0  # weights
    ys = np.arange(h, dtype=np.float32)

    signal_y = np.full(w, np.nan, dtype=np.float32)
    for x in range(w):
        col_bin = binary[:, x]
        idx = np.where(col_bin == 255)[0]
        if idx.size == 0:
            continue
        # weighted centroid using brightness in the inverted image
        weights = inv_float[idx, x]
        wsum = weights.sum()
        if wsum > 1e-6:
            signal_y[x] = (ys[idx] * weights).sum() / wsum
        else:
            # fallback: median position of white pixels
            signal_y[x] = np.median(idx)

    # 5) Fill gaps (optional)
    x_all = np.arange(w, dtype=np.float32)
    valid = ~np.isnan(signal_y)
    if valid.any():
        if interpolate_gaps and (~valid).any():
            signal_y[~valid] = np.interp(x_all[~valid], x_all[valid], signal_y[valid])
    else:
        raise ValueError("No ECG trace detected in this lead (check thresholding/grid suppression).")

    # 6) Convert pixels -> mm -> seconds (X)
    mm_x = x_all / float(pixels_per_mm_x)
    time_sec = mm_x / float(mm_per_sec)

    # 7) Baseline (isoelectric) & amplitude conversion (Y)
    baseline_px = np.nanmedian(signal_y)  # robust baseline estimate
    # Flip sign so upward deflection is positive
    mm_y = -(signal_y - baseline_px) / float(pixels_per_mm_y)
    mv = mm_y * float(mm_per_mv)

    return time_sec, mv

# STEP 6: Save extracted data and plot
def save_excel_and_plot(time, mv, lead_name, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Excel file path
    excel_path = os.path.join(output_folder, f"lead_{lead_name}_data.xlsx")
    
    # Save to Excel using pandas
    df = pd.DataFrame({"Time (s)": time, "Amplitude (mV)": mv})
    df.to_excel(excel_path, index=False)
    
    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(time, mv, label=f"Lead {lead_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"ECG Lead {lead_name}")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"lead_{lead_name}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    return excel_path, plot_path


# MAIN PROCESSING FUNCTION
def process_ecg_pdf_precise(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "ecg_image.png")
    pdf_to_image_fitz(pdf_path, image_path)

    leads_dir = os.path.join(output_dir, "leads")
    results_dir = os.path.join(output_dir, "results")

    lead_images = crop_12_leads_precise(image_path, leads_dir)

    # Use the new grid-based scale detection
    pixels_per_mm_x, pixels_per_mm_y = estimate_pixel_scale_from_grid(lead_images[0][1], debug=True)

    summary = []
    for lead_name, lead_path in lead_images:
        time, mv = extract_signal(lead_path, pixels_per_mm_x, pixels_per_mm_y)
        excel_file, plot_file = save_excel_and_plot(time, mv, lead_name, results_dir)
        summary.append((lead_name, excel_file, plot_file))

    return summary

# EXECUTION BLOCK
if __name__ == "__main__":
    pdf_input_path = "20250620-162409-5.pdf"
    output_dir = r"D:\iisc\Project\ECG_2\Try_2"
    os.makedirs(output_dir, exist_ok=True)
     # Step 1: Convert PDF to image
    image_path = os.path.join(output_dir, "ecg_image.png")
    pdf_to_image_fitz(pdf_input_path, image_path)

    # Step 2: Generate and save grid overlay
    plot_grid_overlay(image_path)
    result = process_ecg_pdf_precise(pdf_input_path, output_dir)
    df_summary = pd.DataFrame(result, columns=["Lead", "Excel File", "Plot File"])

    print(df_summary)
    
