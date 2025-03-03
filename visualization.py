import os
import yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


from config import images_dir_training, labels_dir_training, yaml_path, DEVICE
# Percorsi alle cartelle delle immagini e delle annotazioni


# Carica le informazioni dal file YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

class_names = data.get('names', [])
print("Mapping classi:", class_names)

# Estensioni valide per le immagini
image_extensions = ('.jpg', '.jpeg', '.png')
image_files = sorted([f for f in os.listdir(images_dir_training) if f.lower().endswith(image_extensions)])

# Imposta la dimensione del font per le etichette
font_size = 20
try:
    # Prova a caricare un font TrueType (assicurati che il file esista sul tuo sistema)
    font = ImageFont.truetype("arial.ttf", font_size)
except Exception as e:
    print("Font 'arial.ttf' non trovato, uso font predefinito.")
    font = ImageFont.load_default()

current_index = 0

def load_and_draw(index):
    """Carica l'immagine, disegna i bounding box e le etichette convertendo gli ID in nomi"""
    img_name = image_files[index]
    img_path = os.path.join(images_dir_training, img_name)
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Errore nell'apertura dell'immagine {img_name}: {e}")
        return None

    width, height = img.size

    # Percorso dell'annotazione corrispondente
    annotation_path = os.path.join(labels_dir_training, os.path.splitext(img_name)[0] + '.txt')
    if not os.path.exists(annotation_path):
        print(f"Annotazioni non trovate per: {img_name}")
        return img

    draw = ImageDraw.Draw(img)
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    # Ogni riga è nel formato: classe center_x center_y width height (valori normalizzati)
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id_str, cx, cy, w, h = parts
        try:
            class_idx = int(class_id_str)
        except ValueError:
            class_idx = 0
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        
        # Conversione delle coordinate normalizzate in pixel
        x_min = int((cx - w / 2) * width)
        y_min = int((cy - h / 2) * height)
        x_max = int((cx + w / 2) * width)
        y_max = int((cy + h / 2) * height)
        
        # Disegna il rettangolo
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
        
        # Ottieni il nome della classe se disponibile
        label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        
        # Usa font.getbbox() per ottenere la dimensione del testo
        bbox = font.getbbox(label)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Disegna uno sfondo per l'etichetta
        draw.rectangle([(x_min, y_min - text_height), (x_min + text_width, y_min)], fill="red")
        # Disegna l'etichetta
        draw.text((x_min, y_min - text_height), label,  fill="black", font=font)
    
    return img

# Crea la figura e visualizza la prima immagine
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.2)
displayed_img = load_and_draw(current_index)
im_ax = ax.imshow(displayed_img)
ax.set_title(image_files[current_index])
ax.axis('off')

def on_key(event):
    global current_index, im_ax
    if event.key == 'right':
        current_index = (current_index + 1) % len(image_files)
    elif event.key == 'left':
        current_index = (current_index - 1) % len(image_files)
    else:
        return
    new_img = load_and_draw(current_index)
    if new_img is not None:
        im_ax.set_data(new_img)
        ax.set_title(image_files[current_index])
        fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
