import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import cv2
import fitz  # PyMuPDF
import re
from fpdf import FPDF
import os

# Paths
train_dir = 'C:/Users/Fatim/OneDrive/Documenten/ocrpyto/invoices-4/train'
valid_dir = 'C:/Users/Fatim/OneDrive/Documenten/ocrpyto/invoices-4/validd'

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

# Model Setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save('C:/Users/Fatim/OneDrive/Documenten/ocrpyto/model/mymodely.keras')

print("\nTraining Accuracy and Validation Accuracy per Epoch:")
for epoch, acc in enumerate(history.history['accuracy'], 1):
    val_acc = history.history['val_accuracy'][epoch - 1]
    print(f"Epoch {epoch}: Training Accuracy = {acc:.4f}, Validation Accuracy = {val_acc:.4f}")

# Function to predict image
def predict_image(img_path, model, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    class_labels = {0: 'non-invoice', 1: 'invoice'}
    return class_labels[predicted_class]

# Function to process file
def process_file(file_path):
    if file_path.lower().endswith('.pdf'):
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    else:
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='eng')
        return text

# Function to extract specific data from text
def extract_specific_data(text):
    data = {}
    
    keywords = {
        'order_date': ['Order Date', 'Date', 'Order Date:', 'Date:', 'order date', 'date'],
        'order_number': ['Order Number', 'Order No', 'Order Number:', 'No:', 'order number', 'no'],
        'sold_by': ['Sold By', 'Sold By:', 'Vendor', 'Supplier', 'sold by'],
        'name': ['Name', 'Customer Name', 'Client Name', 'Name:', 'name']
    }
    
    for key, keywords_list in keywords.items():
        for keyword in keywords_list:
            match = re.search(rf'{re.escape(keyword)}\s*[:\-]?\s*(.+)', text, re.IGNORECASE)
            if match:
                data[key] = match.group(1).strip()
                break
        else:
            data[key] = 'Not Found'
    
    return data

# Function to display extracted data
def display_extracted_data():
    clear_window()

    tk.Label(root, text='Extracted Data', font=('Helvetica', 16), bg='pink', fg='white').pack(pady=20)

    for key, value in data.items():
        frame = tk.Frame(root, bg='pink')
        frame.pack(pady=5, padx=10, fill='x')

        label = tk.Label(frame, text=f"{key.replace('_', ' ').title()}: {value}", bg='pink', fg='white', font=('Helvetica', 12))
        label.pack(side='left')

    tk.Button(root, text='Push Data to Form', command=show_data_form, bg='#ffb6c1', fg='white').pack(pady=10)

# Function to show data form
def show_data_form():
    clear_window()

    tk.Label(root, text='Extracted Data Form', font=('Helvetica', 16), bg='pink', fg='white').pack(pady=20)

    global entries
    entries = {}
    for key, value in data.items():
        frame = tk.Frame(root, bg='pink')
        frame.pack(pady=5, padx=10, fill='x')

        label = tk.Label(frame, text=f"{key.replace('_', ' ').title()}:", bg='pink', fg='white', font=('Helvetica', 12))
        label.pack(side='left')

        entry = tk.Entry(frame, font=('Helvetica', 12))
        entry.insert(0, value)
        entry.pack(side='left', padx=5)
        entries[key] = entry

    tk.Button(root, text='Save', command=save_form, bg='#ffb6c1', fg='white').pack(pady=10)
    tk.Button(root, text='Create PDF', command=create_pdf, bg='#ffb6c1', fg='white').pack(pady=10)

# Function to save form data
def save_form():
    directory = r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, 'form_data.txt')
    with open(file_path, 'w') as file:
        for key, entry in entries.items():
            file.write(f"{key.replace('_', ' ').title()}: {entry.get()}\n")

    messagebox.showinfo('Form Saved', f'Form data has been saved successfully at {file_path}.')

# Function to create PDF
def create_pdf():
    directory = r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, 'extracted_data.pdf')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    for key, entry in entries.items():
        pdf.cell(0, 10, f"{key.replace('_', ' ').title()}: {entry.get()}", ln=True)
    
    pdf.output(file_path)

    messagebox.showinfo('PDF Created', f'PDF has been created successfully at {file_path}.')

# Function to clear window
def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

# Function to handle file upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("All Files", "*.*")])
    if not file_path:
        return
    
    result = predict_image(file_path, model)
    
    if result == 'invoice':
        # Proceed with processing and extracting data without showing the message
        text = process_file(file_path)
        print("Extracted Text:\n", text)  # Debugging: Print the full text extracted

        # Extract and display specific data
        global data
        data = extract_specific_data(text)
        display_extracted_data()
    else:
        # Show message if image is not classified as an invoice
        messagebox.showinfo("Prediction Result", "The image is not an invoice.")

    # Display the image in the panel regardless of classification
    img = Image.open(file_path)
    img.thumbnail((250, 250))
    img = ImageTk.PhotoImage(img)
    
    panel.config(image=img)
    panel.image = img

# Class for OCR application
class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Fatimah OCR Application')
        self.root.configure(bg='pink')
        self.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        self.data = {}
        self.show_welcome_page()

    def show_welcome_page(self):
        clear_window()
        tk.Label(self.root, text='Welcome to Fatimah OCR Application', font=('Helvetica', 16), bg='pink', fg='white').pack(pady=20)
        tk.Button(self.root, text='Upload to Document', command=self.show_upload_page, bg='#ffb6c1', fg='white').pack(pady=10)

    def show_upload_page(self):
        clear_window()
        tk.Label(self.root, text='Drag and drop files here or upload from', font=('Helvetica', 14), bg='pink', fg='white').pack(pady=20)

        options = {
            'My Device': r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto\device_icon.png',
            'Library': r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto\library_icon.png',
            'Link': r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto\link_icon.png',
            'Camera': r'C:\Users\Fatim\OneDrive\Documenten\ocrpyto\camera_icon.png'
        }

        for text, icon_path in options.items():
            try:
                img = Image.open(icon_path)
                img = img.resize((50, 50))
                photo = ImageTk.PhotoImage(img)

                frame = tk.Frame(self.root, bg='pink', width=100, height=150)
                frame.pack_propagate(False)
                frame.pack(pady=10, padx=10, side='left', anchor='n')

                icon_label = tk.Label(frame, image=photo, bg='pink')
                icon_label.image = photo
                icon_label.pack(pady=(10, 5))

                text_label = tk.Label(frame, text=text, bg='pink', fg='white', font=('Helvetica', 12))
                text_label.pack()

                frame.bind('<Button-1>', lambda e, t=text: self.create_command(t)())
                frame.bind('<Enter>', lambda e: frame.config(cursor='hand2'))
                frame.bind('<Leave>', lambda e: frame.config(cursor=''))

            except Exception as e:
                print(f"Error loading image {icon_path}: {e}")

    def create_command(self, text):
        def command():
            if text == 'My Device':
                upload_and_predict()
            elif text == 'Library':
                messagebox.showinfo('Library Upload', 'This feature is not implemented yet.')
            elif text == 'Link':
                messagebox.showinfo('Link Upload', 'This feature is not implemented yet.')
            elif text == 'Camera':
                messagebox.showinfo('Camera Upload', 'This feature is not implemented yet.')
        return command

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
