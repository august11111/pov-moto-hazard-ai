import os
import jsonlines
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm

def wrap_text(text, width=80):
    """Wrap text into lines no longer than `width` characters."""
    if not text:
        return ["N/A"]
    words = text.split()
    lines = []
    line = ""
    for word in words:
        if len(line) + len(word) + 1 <= width:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    if line:
        lines.append(line.strip())
    return lines

def create_report_pdf(results_path, frames_folder, output_path="frame_report.pdf"):
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4

    with jsonlines.open(results_path) as reader:
        for frame in reader:
            img_path = os.path.join(frames_folder, os.path.basename(frame["image_path"]))
            if not os.path.exists(img_path):
                print(f"Image not found: {img_path}")
                continue

            # Draw image centered
            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
                aspect = img_width / img_height
                max_img_width = width - 4 * cm
                max_img_height = height / 2.2
                target_height = min(max_img_height, max_img_width / aspect)
                target_width = target_height * aspect

                x_img = (width - target_width) / 2
                y_img = height - target_height - 2.5 * cm
                img_reader = ImageReader(image)
                c.drawImage(img_reader, x_img, y_img, width=target_width, height=target_height)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

            # Text formatting
            y = y_img - 1.2 * cm
            text_margin = 3 * cm

            c.setFont("Helvetica-Bold", 12)
            c.drawString(text_margin, y, f"Frame ID: {frame['frame_id']} | Timestamp: {frame['timestamp']:.2f}s")

            y -= 1.2 * cm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(text_margin, y, "Final Advice:")
            y -= 0.7 * cm
            c.setFont("Helvetica", 10)
            for line in wrap_text(frame.get("final_advice", "N/A")):
                c.drawString(text_margin, y, line)
                y -= 0.5 * cm

            y -= 0.7 * cm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(text_margin, y, "Global Advice:")
            y -= 0.7 * cm
            c.setFont("Helvetica", 10)
            for line in wrap_text(frame.get("global_advice", "N/A")):
                c.drawString(text_margin, y, line)
                y -= 0.5 * cm

            y -= 0.7 * cm
            c.setFont("Helvetica-Bold", 11)
            override = "Yes" if frame.get("is_override") else "No"
            c.drawString(text_margin, y, f"Override triggered: {override}")

            c.showPage()

    c.save()
    print(f"✅ PDF saved to: {output_path}")

# === Run the generator ===
if __name__ == "__main__":
    results_path = os.environ.get("RESULTS_PATH", "results/enhanced_results.jsonl")
    frames_folder = os.environ.get("FRAMES_FOLDER", "results/frames")
    output_path = os.environ.get("OUTPUT_PATH", "report/advice_report.pdf")

    create_report_pdf(
        results_path=results_path,
        frames_folder=frames_folder,
        output_path=output_path
    )

