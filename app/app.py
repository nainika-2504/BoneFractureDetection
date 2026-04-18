
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.units import inch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(2048, 2)
    )
    model.load_state_dict(torch.load(
        "best_bone_model_v2.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model             = load_model()
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
feature_extractor.eval()
mean_features     = np.load("ood_mean.npy")
cov_inv           = np.load("ood_cov_inv.npy")
ood_threshold     = np.load("ood_threshold.npy")[0]
class_names       = ["fracture", "normal"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def mahalanobis_distance(tensor):
    with torch.no_grad():
        feat = feature_extractor(tensor)
        feat = feat.squeeze(-1).squeeze(-1).cpu().numpy()[0]
    diff = feat - mean_features
    return float(np.sqrt(diff @ cov_inv @ diff))

def get_severity(confidence):
    if confidence >= 95:
        return "SEVERE",   "Immediate medical attention required"
    elif confidence >= 80:
        return "MODERATE", "Medical consultation recommended soon"
    else:
        return "MILD",     "Monitor and consult a doctor"

def get_location(grayscale_cam):
    h, w   = grayscale_cam.shape
    top    = grayscale_cam[:h//3, :].mean()
    middle = grayscale_cam[h//3:2*h//3, :].mean()
    bottom = grayscale_cam[2*h//3:, :].mean()
    left   = grayscale_cam[:, :w//2].mean()
    right  = grayscale_cam[:, w//2:].mean()
    vert   = max({"Upper region": top, "Middle region": middle,
                  "Lower region": bottom}, key=lambda k: {"Upper region": top,
                  "Middle region": middle, "Lower region": bottom}[k])
    horiz  = "Left side" if left > right else "Right side"
    return f"{vert}, {horiz}"

def get_recommendations(prediction, severity):
    if prediction == "normal":
        return ["No fracture detected",
                "Continue regular activities",
                "Monitor for any pain or discomfort",
                "Consult doctor if symptoms develop"]
    recs = ["DISCLAIMER: AI suggestion only - consult a medical professional"]
    if severity == "SEVERE":
        recs += ["Seek immediate emergency medical care",
                 "Immobilize the affected area",
                 "Do not apply weight or pressure",
                 "Surgery may be required - doctor evaluation needed"]
    elif severity == "MODERATE":
        recs += ["Visit orthopedic specialist within 24-48 hours",
                 "Apply ice pack to reduce swelling",
                 "Keep affected area elevated",
                 "Casting or splinting likely required"]
    else:
        recs += ["Schedule doctor appointment within 1 week",
                 "Rest the affected area",
                 "Apply ice pack for 20 mins every 2-3 hours",
                 "Physiotherapy may be recommended"]
    return recs

def generate_gradcam(img, tensor, pred):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    target_layer  = [model.layer4[-1]]
    cam           = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=tensor,
                        targets=[ClassifierOutputTarget(pred)])[0]
    rgb_img = np.array(img.resize((224,224))).astype(np.float32) / 255.0
    viz     = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return viz, grayscale_cam

def generate_pdf(patient_name, patient_age, prediction, confidence,
                 severity, location, recommendations,
                 orig_path, gradcam_path):
    output = f"/tmp/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    doc    = SimpleDocTemplate(output, pagesize=A4,
                               rightMargin=40, leftMargin=40,
                               topMargin=40,  bottomMargin=40)
    story  = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("t", fontSize=20,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a237e"), spaceAfter=5)
    sub_style   = ParagraphStyle("s", fontSize=11,
        fontName="Helvetica", textColor=colors.grey, spaceAfter=15)
    sec_style   = ParagraphStyle("se", fontSize=13,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a237e"), spaceAfter=8)
    norm_style  = ParagraphStyle("n", fontSize=10,
        fontName="Helvetica", spaceAfter=4)
    disc_style  = ParagraphStyle("d", fontSize=8,
        fontName="Helvetica-Oblique",
        textColor=colors.red, spaceAfter=4)

    story.append(Paragraph("Bone Fracture Detection Report", title_style))
    story.append(Paragraph("AI-Assisted Medical Imaging Analysis", sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Patient Information", sec_style))
    pt = Table([
        ["Patient Name", patient_name],
        ["Age",          f"{patient_age} years"],
        ["Report Date",  datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Report ID",    f"BF-{datetime.now().strftime('%Y%m%d%H%M%S')}"]
    ], colWidths=[2*inch, 4*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8eaf6")),
        ("FONTNAME",   (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    story.append(pt)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Analysis Result", sec_style))
    result_text = "FRACTURE DETECTED" if prediction == "fracture" else "NORMAL BONE"
    rt = Table([
        ["Prediction", result_text],
        ["Confidence", f"{confidence:.2f}%"],
        ["Severity",   severity if prediction == "fracture" else "N/A"],
        ["Location",   location if prediction == "fracture" else "N/A"],
    ], colWidths=[2*inch, 4*inch])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0,0),  (0,-1),  colors.HexColor("#e8eaf6")),
        ("BACKGROUND", (1,0),  (1,0),
         colors.red if prediction == "fracture" else colors.green),
        ("TEXTCOLOR",  (1,0),  (1,0),   colors.white),
        ("FONTNAME",   (0,0),  (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),  (-1,-1), 10),
        ("GRID",       (0,0),  (-1,-1), 0.5, colors.grey),
        ("PADDING",    (0,0),  (-1,-1), 6),
    ]))
    story.append(rt)
    story.append(Spacer(1, 15))

    story.append(Paragraph("X-ray Analysis", sec_style))
    img_table = Table([[
        RLImage(orig_path,    width=2.5*inch, height=2.5*inch),
        RLImage(gradcam_path, width=2.5*inch, height=2.5*inch)
    ],[
        Paragraph("Original X-ray",   norm_style),
        Paragraph("Grad-CAM Heatmap", norm_style)
    ]], colWidths=[3*inch, 3*inch])
    img_table.setStyle(TableStyle([
        ("ALIGN",   (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 10),
        ("BOX",     (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("Recommendations", sec_style))
    for i, rec in enumerate(recommendations):
        if i == 0 and prediction == "fracture":
            story.append(Paragraph(f"{rec}", disc_style))
        else:
            story.append(Paragraph(f"• {rec}", norm_style))

    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI model and is NOT "
        "a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for medical decisions.",
        disc_style))
    doc.build(story)
    return output

def predict(image, patient_name, patient_age):
    if image is None:
        return "Please upload an image", None, None, None

    if not patient_name:
        patient_name = "Unknown"
    if not patient_age:
        patient_age  = "N/A"

    img    = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # OOD check
    dist = mahalanobis_distance(tensor)
    if dist > ood_threshold:
        return "REJECTED - Not a bone X-ray!\nPlease upload a valid bone X-ray.",                {"Not a bone X-ray": 1.0}, None, None

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        pred    = outputs.argmax(1).item()
        conf    = probs[0][pred].item() * 100

    prediction    = class_names[pred]
    gradcam_viz, grayscale_cam = generate_gradcam(img, tensor, pred)

    severity      = get_severity(conf)[0] if prediction == "fracture" else "N/A"
    location      = get_location(grayscale_cam) if prediction == "fracture" else "N/A"
    recs          = get_recommendations(prediction, severity)

    # Save images for PDF
    orig_path    = "/tmp/orig.png"
    gradcam_path = "/tmp/gradcam.png"
    img.resize((224,224)).save(orig_path)
    Image.fromarray(gradcam_viz).save(gradcam_path)

    # Generate PDF
    pdf_path = generate_pdf(
        patient_name, patient_age, prediction, conf,
        severity, location, recs, orig_path, gradcam_path
    )

    label  = "FRACTURE DETECTED" if pred == 0 else "NORMAL BONE"
    result = f"{label}\nConfidence: {conf:.2f}%\nSeverity: {severity}\nLocation: {location}"
    confs  = {"Fracture": round(probs[0][0].item(), 3),
               "Normal"  : round(probs[0][1].item(), 3)}

    return result, confs, gradcam_viz, pdf_path

with gr.Blocks(title="Bone Fracture Detection") as demo:
    gr.Markdown("""
    # Bone Fracture Detection
    ### AI-Assisted Medical Imaging Analysis
    > Upload a bone X-ray to get instant analysis + downloadable PDF report
    ---
    """)
    with gr.Row():
        with gr.Column():
            image_input   = gr.Image(label="Upload Bone X-ray")
            patient_name  = gr.Textbox(label="Patient Name", placeholder="Enter patient name")
            patient_age   = gr.Textbox(label="Patient Age",  placeholder="Enter patient age")
            submit_btn    = gr.Button("Analyze & Generate Report", variant="primary")
        with gr.Column():
            result_text   = gr.Textbox(label="Analysis Result", lines=5)
            confidence    = gr.Label(label="Confidence Scores")
            gradcam_out   = gr.Image(label="Grad-CAM Heatmap")
            pdf_output    = gr.File(label="Download PDF Report")

    submit_btn.click(
        fn=predict,
        inputs=[image_input, patient_name, patient_age],
        outputs=[result_text, confidence, gradcam_out, pdf_output]
    )
    gr.Markdown("""
    ---
    **Model**: ResNet50 | **Accuracy**: 97.37% | **OOD Detection**: Enabled
    > This tool is for educational purposes only. Not a substitute for medical advice.
    """)

demo.launch()
