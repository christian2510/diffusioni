import customtkinter as ctk
import threading
import io
from PIL import Image, ImageTk # Benötigt Pillow: pip install Pillow
import os
from datetime import datetime
import torch
import numpy as np # Benötigt numpy: pip install numpy
# Benötigt diffusers: pip install diffusers transformers accelerate safetensors
# Benötigt Scheduler: pip install --upgrade diffusers
# Optional für GPU-Optimierung: pip install xformers
# Optional für 8-Bit-Quantisierung: pip install bitsandbytes
# Triton ist nicht mehr zwingend erforderlich, da torch.compile deaktiviert wird, aber safetensors ist weiterhin nötig
import safetensors.torch # Für das Auslesen von Safetensors-Metadaten
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DDIMScheduler,
    DDPMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
    DPMSolverSDEScheduler,
)
from tkinter import filedialog, messagebox # Importiere filedialog und messagebox für Dateiauswahl und Bestätigungsdialoge
import random # Für zufällige Seeds
import json # Für das Speichern von Metadaten
import traceback # Importiere traceback für detaillierte Fehlerausgaben
from collections import deque # Für den Prompt-Verlauf
import sys # Neu: Für Kommandozeilenargumente
import time # Neu: Für Zeitmessung
import gc # Neu: Für Garbage Collection

try:
    import pyperclip # Für Zwischenablage-Operationen
except ImportError:
    pyperclip = None # Fallback, wenn pyperclip nicht installiert ist

# Setzt das Erscheinungsbild (System, Light, Dark)
ctk.set_appearance_mode("Dark")
# Setzt das Standard-Farbschema (blue, dark-blue, green)
ctk.set_default_color_theme("blue")

# Verzeichnis für gespeicherte Bilder und Metadaten
IMAGE_DIR = "output" # Geändert von "generated_images_local" zu "output"
METADATA_FILE = os.path.join(IMAGE_DIR, "image_data_local.json")
PROMPT_HISTORY_FILE = os.path.join(IMAGE_DIR, "prompt_history.json")
MODELS_DIR = "models" # Neues Verzeichnis für Modelldateien

class ImageGeneratorApp(ctk.CTk):
    """
    Hauptanwendungsklasse für den KI-Bildgenerator mit lokaler Stable Diffusion.
    """
    def __init__(self, force_cpu=False): # Neu: force_cpu Parameter
        super().__init__()

        # Neu: CPU-Modus erzwingen und Gerät festlegen
        self.force_cpu = force_cpu
        self.device = "cpu" if self.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_status_message = f"Bereit. Gerät: {self.device.upper()}"
        if self.device == "cpu" and not self.force_cpu:
            self.initial_status_message += " (Keine GPU gefunden)"
        elif self.force_cpu:
            self.initial_status_message += " (CPU-Modus erzwungen)"

        self.title(f"Diffusioni v.0.1 Alpha ({self.device.upper()} Modus)") # Neuer Titel mit Modus
        self.geometry("1400x900") # Angepasste Größe für Zwei-Spalten-Layout
        self.minsize(1000, 750) # Mindestgröße angepasst

        # Stellt sicher, dass das Bildverzeichnis existiert
        try:
            os.makedirs(IMAGE_DIR, exist_ok=True)
            print(f"DEBUG: Bildverzeichnis '{os.path.abspath(IMAGE_DIR)}' existiert oder wurde erstellt.")
        except OSError as e:
            messagebox.showerror("Fehler beim Erstellen des Verzeichnisses", f"Konnte das Bildverzeichnis nicht erstellen: {IMAGE_DIR}\nBitte überprüfen Sie die Berechtigungen oder wählen Sie einen anderen Speicherort.\nFehler: {e}")
            print(f"ERROR: Fehler beim Erstellen des Bildverzeichnisses: {e}")
            # Programm könnte hier beendet werden, wenn das Verzeichnis unerlässlich ist
            # self.destroy() 

        # Stellt sicher, dass der Modelle-Ordner existiert
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            print(f"DEBUG: Modelle-Verzeichnis '{os.path.abspath(MODELS_DIR)}' existiert oder wurde erstellt.")
        except OSError as e:
            messagebox.showerror("Fehler beim Erstellen des Modelle-Verzeichnisses", f"Konnte das Modelle-Verzeichnis nicht erstellen: {MODELS_DIR}\nBitte überprüfen Sie die Berechtigungen oder wählen Sie einen anderen Speicherort.\nFehler: {e}")
            print(f"ERROR: Fehler beim Erstellen des Modelle-Verzeichnisses: {e}")


        # Konfiguriert das Gitter für das Hauptfenster
        self.grid_columnconfigure(0, weight=0) # Linke Spalte (fest/weniger Gewicht)
        self.grid_columnconfigure(1, weight=1) # Rechte Spalte (nimmt den Rest des Platzes ein)
        self.grid_rowconfigure(0, weight=1) # Nur eine Zeile für den Hauptinhalt

        # Prompt-Verlauf (z.B. die letzten 10 Prompts)
        self.prompt_history = deque(maxlen=10)
        # _load_prompt_history wird jetzt später aufgerufen, nachdem das Widget erstellt wurde.

        # Event, um den Generierungs-Thread zu stoppen
        self.stop_event = threading.Event()
        # Bindet die on_closing-Methode an das Schließen des Fensters
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


        # --- Linke Spalte: Eingabebereich und Einstellungen ---
        self.left_panel = ctk.CTkFrame(self, corner_radius=12, fg_color=("gray85", "gray15"))
        self.left_panel.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.left_panel.grid_columnconfigure(0, weight=1) # Eine Spalte im linken Panel, die sich ausdehnt
        self.left_panel.grid_rowconfigure(12, weight=1) # Macht Platz für den unteren Teil

        # Modell-Auswahl (ersetzt Pfadeingabe und Durchsuchen-Button)
        self.model_selection_label = ctk.CTkLabel(self.left_panel, text="Verfügbare Modelle (.safetensors im /models Ordner):", font=ctk.CTkFont(size=15, weight="bold"))
        self.model_selection_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")
        
        self.model_optionmenu = ctk.CTkOptionMenu(self.left_panel, values=["Keine Modelle gefunden"], command=self._on_model_select, corner_radius=8)
        self.model_optionmenu.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="ew")
        
        # Initialisiere die Liste der Modelle beim Start (jetzt nach der Definition des Widgets)
        # self._populate_model_list() # Dies wird an das Ende von __init__ verschoben

        self.load_model_button = ctk.CTkButton(self.left_panel, text="Modell laden", command=self.load_model, height=35, corner_radius=8)
        self.load_model_button.grid(row=4, column=1, padx=(0, 20), pady=(0, 15), sticky="e") # Angepasste Zeile

        # Checkbox für SDXL-Modell
        self.is_sdxl_checkbox = ctk.CTkCheckBox(self.left_panel, text="SDXL-Modell laden (automatisch erkannt)", font=ctk.CTkFont(size=13))
        self.is_sdxl_checkbox.grid(row=2, column=0, padx=20, pady=(5, 5), sticky="w")
        self.sdxl_info_label = ctk.CTkLabel(self.left_panel, text="(Benötigt oft mehr VRAM/RAM)", font=ctk.CTkFont(size=10), text_color="gray")
        self.sdxl_info_label.grid(row=2, column=0, padx=(220, 0), pady=(5, 5), sticky="w")

        # Checkbox für 8-Bit-Quantisierung
        self.quantization_checkbox = ctk.CTkCheckBox(self.left_panel, text="8-Bit-Quantisierung aktivieren (nur GPU)", font=ctk.CTkFont(size=13), command=self._toggle_quantization_info)
        self.quantization_checkbox.grid(row=3, column=0, padx=20, pady=(5, 15), sticky="w")
        self.quantization_info_label = ctk.CTkLabel(self.left_panel, text="(Reduziert VRAM, macht langsamer)", font=ctk.CTkFont(size=10), text_color="gray")
        self.quantization_info_label.grid(row=3, column=0, padx=(220, 0), pady=(5, 15), sticky="w")
        if not torch.cuda.is_available() or self.force_cpu: # Deaktivieren, wenn keine GPU oder CPU-Modus erzwungen
            self.quantization_checkbox.configure(state="disabled")
            self.quantization_info_label.configure(text="(Nur für NVIDIA GPUs verfügbar)")


        self.prompt_label = ctk.CTkLabel(self.left_panel, text="Bildbeschreibung (Prompt):", font=ctk.CTkFont(size=16, weight="bold"))
        self.prompt_label.grid(row=5, column=0, padx=20, pady=(15, 5), sticky="w")

        self.prompt_entry = ctk.CTkEntry(self.left_panel, placeholder_text="Eine Katze im Astronautenanzug auf dem Mond", height=40, corner_radius=8)
        self.prompt_entry.grid(row=6, column=0, padx=20, pady=(0, 15), sticky="ew")
        self.prompt_entry.bind("<Return>", self.generate_image_event)
        self.prompt_entry.configure(state="disabled")

        self.clear_prompt_button = ctk.CTkButton(self.left_panel, text="Prompt leeren", command=self._clear_prompt, height=40, corner_radius=8)
        self.clear_prompt_button.grid(row=6, column=1, padx=(0, 20), pady=(0, 15), sticky="e")
        self.clear_prompt_button.configure(state="disabled")

        self.negative_prompt_label = ctk.CTkLabel(self.left_panel, text="Negativer Prompt (was nicht im Bild sein soll):", font=ctk.CTkFont(size=14))
        self.negative_prompt_label.grid(row=7, column=0, padx=20, pady=(0, 5), sticky="w")

        self.negative_prompt_entry = ctk.CTkEntry(self.left_panel, placeholder_text="schlecht gezeichnet, unschön, Text, Signatur", height=40, corner_radius=8)
        self.negative_prompt_entry.grid(row=8, column=0, padx=20, pady=(0, 15), sticky="ew")
        self.negative_prompt_entry.configure(state="disabled")

        self.generate_button = ctk.CTkButton(self.left_panel, text="Bild generieren", command=self.generate_image_event, height=40, corner_radius=8, font=ctk.CTkFont(size=15, weight="bold"))
        self.generate_button.grid(row=8, column=1, padx=(0, 20), pady=(0, 15), sticky="e")
        self.generate_button.configure(state="disabled")

        # Prompt-Verlauf Dropdown
        self.prompt_history_label = ctk.CTkLabel(self.left_panel, text="Prompt-Verlauf:", font=ctk.CTkFont(size=14))
        self.prompt_history_label.grid(row=9, column=0, padx=20, pady=(0, 5), sticky="w")
        self.prompt_history_optionmenu = ctk.CTkOptionMenu(self.left_panel, values=["Kein Verlauf"], command=self._load_prompt_from_history, corner_radius=8)
        self.prompt_history_optionmenu.grid(row=10, column=0, columnspan=2, padx=20, pady=(0, 15), sticky="ew")
        self._load_prompt_history() # HIERHER VERSCHOBEN


        # --- Einstellungen für die Bildgenerierung ---
        self.settings_frame = ctk.CTkFrame(self.left_panel, corner_radius=12, fg_color=("gray85", "gray15"))
        self.settings_frame.grid(row=11, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew")
        self.settings_frame.grid_columnconfigure((0,1,2,3), weight=1)

        self.settings_label = ctk.CTkLabel(self.settings_frame, text="Generierungs-Einstellungen:", font=ctk.CTkFont(size=15, weight="bold"))
        self.settings_label.grid(row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w")

        # Bildgröße
        self.width_options = ["512", "768", "1024"]
        self.height_options = ["512", "768", "1024"]

        self.width_label = ctk.CTkLabel(self.settings_frame, text="Breite:", font=ctk.CTkFont(size=13))
        self.width_label.grid(row=1, column=0, padx=(15, 5), pady=(5, 0), sticky="w")
        self.width_optionmenu = ctk.CTkOptionMenu(self.settings_frame, values=self.width_options, command=self._update_size_options, corner_radius=8)
        self.width_optionmenu.set("512")
        self.width_optionmenu.grid(row=2, column=0, padx=(15, 5), pady=(0, 10), sticky="ew")
        self.width_optionmenu.configure(state="disabled")

        self.height_label = ctk.CTkLabel(self.settings_frame, text="Höhe:", font=ctk.CTkFont(size=13))
        self.height_label.grid(row=1, column=1, padx=(5, 15), pady=(5, 0), sticky="w")
        self.height_optionmenu = ctk.CTkOptionMenu(self.settings_frame, values=self.height_options, command=self._update_size_options, corner_radius=8)
        self.height_optionmenu.set("512")
        self.height_optionmenu.grid(row=2, column=1, padx=(5, 15), pady=(0, 10), sticky="ew")
        self.height_optionmenu.configure(state="disabled")

        # Steps
        self.steps_label = ctk.CTkLabel(self.settings_frame, text="Schritte (Steps):", font=ctk.CTkFont(size=13))
        self.steps_label.grid(row=3, column=0, padx=(15, 5), pady=(5, 0), sticky="w")
        self.steps_slider = ctk.CTkSlider(self.settings_frame, from_=10, to=100, number_of_steps=90, command=self._update_steps_label, corner_radius=8)
        self.steps_slider.set(30)
        self.steps_slider.grid(row=4, column=0, padx=(15, 5), pady=(0, 10), sticky="ew")
        self.steps_slider.configure(state="disabled")
        self.steps_value_label = ctk.CTkLabel(self.settings_frame, text=f"{int(self.steps_slider.get())}")
        self.steps_value_label.grid(row=4, column=1, padx=(0, 15), pady=(0, 10), sticky="w")

        # CFG Scale
        self.cfg_label = ctk.CTkLabel(self.settings_frame, text="CFG-Skala:", font=ctk.CTkFont(size=13))
        self.cfg_label.grid(row=3, column=2, padx=(15, 5), pady=(5, 0), sticky="w")
        self.cfg_slider = ctk.CTkSlider(self.settings_frame, from_=1.0, to=20.0, number_of_steps=190, command=self._update_cfg_label, corner_radius=8)
        self.cfg_slider.set(7.5)
        self.cfg_slider.grid(row=4, column=2, padx=(15, 5), pady=(0, 10), sticky="ew")
        self.cfg_slider.configure(state="disabled")
        self.cfg_value_label = ctk.CTkLabel(self.settings_frame, text=f"{self.cfg_slider.get():.1f}")
        self.cfg_value_label.grid(row=4, column=3, padx=(0, 15), pady=(0, 10), sticky="w")

        # Seed
        self.seed_label = ctk.CTkLabel(self.settings_frame, text="Seed:", font=ctk.CTkFont(size=13))
        self.seed_label.grid(row=5, column=0, padx=(15, 5), pady=(5, 0), sticky="w")
        self.seed_entry = ctk.CTkEntry(self.settings_frame, placeholder_text="-1 für Zufall", corner_radius=8)
        self.seed_entry.grid(row=6, column=0, padx=(15, 5), pady=(0, 15), sticky="ew")
        self.seed_entry.insert(0, "-1") # Standard auf Zufall
        self.seed_entry.configure(state="disabled")
        self.random_seed_button = ctk.CTkButton(self.settings_frame, text="Zufälliger Seed", command=self._set_random_seed, corner_radius=8)
        self.random_seed_button.grid(row=6, column=1, padx=(5, 15), pady=(0, 15), sticky="w")
        self.random_seed_button.configure(state="disabled")

        # Scheduler
        self.scheduler_options = [
            "Euler",
            "Euler Ancestral",
            "DPM++ 2M",
            "DPM++ 2M Karras",
            "DPM++ SDE",
            "DPM++ SDE Karras",
            "LMS",
            "LMS Karras",
            "DDIM",
            "PNDM",
            "DDPM",
            "Heun",
            "KDPM2",
            "KDPM2 Ancestral",
            "DEIS",
            "UniPC"
        ]
        self.scheduler_map = { # Mapping von Namen zu Scheduler-Klassen
            "Euler": EulerDiscreteScheduler,
            "Euler Ancestral": EulerAncestralDiscreteScheduler,
            "DPM++ 2M": DPMSolverMultistepScheduler,
            "DPM++ 2M Karras": DPMSolverMultistepScheduler,
            "DPM++ SDE": DPMSolverSDEScheduler,
            "DPM++ SDE Karras": DPMSolverSDEScheduler,
            "LMS": LMSDiscreteScheduler,
            "LMS Karras": LMSDiscreteScheduler,
            "DDIM": DDIMScheduler,
            "PNDM": PNDMScheduler,
            "DDPM": DDPMScheduler,
            "Heun": HeunDiscreteScheduler,
            "KDPM2": KDPM2DiscreteScheduler,
            "KDPM2 Ancestral": KDPM2AncestralDiscreteScheduler,
            "DEIS": DEISMultistepScheduler,
            "UniPC": UniPCMultistepScheduler,
        }
        self.scheduler_label = ctk.CTkLabel(self.settings_frame, text="Scheduler:", font=ctk.CTkFont(size=13))
        self.scheduler_label.grid(row=5, column=2, padx=(15, 5), pady=(5, 0), sticky="w")
        self.scheduler_optionmenu = ctk.CTkOptionMenu(self.settings_frame, values=self.scheduler_options, corner_radius=8)
        self.scheduler_optionmenu.set("Euler") # Standard-Scheduler
        self.scheduler_optionmenu.grid(row=6, column=2, padx=(15, 5), pady=(0, 15), sticky="ew")
        self.scheduler_optionmenu.configure(state="disabled")

        # --- Eigene Größe Eingabefelder ---
        self.custom_size_label = ctk.CTkLabel(self.settings_frame, text="Eigene Größe (B x H):", font=ctk.CTkFont(size=13))
        self.custom_size_label.grid(row=7, column=0, padx=(15, 5), pady=(5, 0), sticky="w")
        self.custom_width_entry = ctk.CTkEntry(self.settings_frame, placeholder_text="512", width=70, corner_radius=8)
        self.custom_width_entry.grid(row=8, column=0, padx=(15, 5), pady=(0, 15), sticky="w")
        self.custom_width_entry.configure(state="disabled")

        self.custom_height_entry = ctk.CTkEntry(self.settings_frame, placeholder_text="512", width=70, corner_radius=8)
        self.custom_height_entry.grid(row=8, column=1, padx=(5, 15), pady=(0, 15), sticky="w")
        self.custom_height_entry.configure(state="disabled")

        self.use_custom_size_checkbox = ctk.CTkCheckBox(self.settings_frame, text="Eigene Größe verwenden", command=self._toggle_custom_size, font=ctk.CTkFont(size=13))
        self.use_custom_size_checkbox.grid(row=7, column=1, columnspan=2, padx=(5,15), pady=(5,0), sticky="w")
        self.use_custom_size_checkbox.configure(state="disabled")

        # --- Anzahl der Bilder ---
        self.num_images_label = ctk.CTkLabel(self.settings_frame, text="Anzahl Bilder:", font=ctk.CTkFont(size=13))
        self.num_images_label.grid(row=7, column=2, padx=(15, 5), pady=(5, 0), sticky="w")
        self.num_images_entry = ctk.CTkEntry(self.settings_frame, placeholder_text="1", width=50, corner_radius=8)
        self.num_images_entry.grid(row=8, column=2, padx=(15, 5), pady=(0, 15), sticky="w")
        self.num_images_entry.insert(0, "1")
        self.num_images_entry.configure(state="disabled")

        # --- Live-Vorschau (entfernt, da es Generierung stark verlangsamt) ---
        # self.live_preview_checkbox = ctk.CTkCheckBox(self.settings_frame, text="Live-Vorschau anzeigen (verlangsamt Generierung)", font=ctk.CTkFont(size=13))
        # self.live_preview_checkbox.grid(row=9, column=0, columnspan=4, padx=15, pady=(5, 15), sticky="w")
        # self.live_preview_checkbox.configure(state="disabled")


        # --- Rechte Spalte: Bildanzeigebereich, Details und Buttons ---
        self.right_panel = ctk.CTkFrame(self, corner_radius=12, fg_color=("gray85", "gray15"))
        self.right_panel.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(0, weight=1) # Bildlabel nimmt den meisten Platz ein

        self.image_label = ctk.CTkLabel(self.right_panel, text="Hier erscheint Ihr generiertes Bild.\n\nBitte laden Sie zuerst ein Modell.", font=ctk.CTkFont(size=18), text_color="gray", fg_color="transparent")
        self.image_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew") # Zusätzliche Polsterung um das Bild


        # --- Details zum generierten Bild ---
        self.image_details_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.image_details_frame.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew") # Angepasste Polsterung
        self.image_details_frame.grid_columnconfigure(0, weight=1)

        self.details_prompt_label = ctk.CTkLabel(self.image_details_frame, text="Prompt: ", wraplength=700, justify="left", font=ctk.CTkFont(size=12, weight="bold"))
        self.details_prompt_label.grid(row=0, column=0, padx=5, pady=0, sticky="w")
        self.details_negative_prompt_label = ctk.CTkLabel(self.image_details_frame, text="Negativ: ", wraplength=700, justify="left", font=ctk.CTkFont(size=10), text_color="gray")
        self.details_negative_prompt_label.grid(row=1, column=0, padx=5, pady=0, sticky="w")
        self.details_params_label = ctk.CTkLabel(self.image_details_frame, text="Parameter: ", font=ctk.CTkFont(size=10), text_color="gray")
        self.details_params_label.grid(row=2, column=0, padx=5, pady=0, sticky="w")
        self.details_generation_time_label = ctk.CTkLabel(self.image_details_frame, text="Dauer: ", font=ctk.CTkFont(size=10), text_color="gray") # Neu: Label für Generierungsdauer
        self.details_generation_time_label.grid(row=3, column=0, padx=5, pady=0, sticky="w")

        self.copy_prompt_button = ctk.CTkButton(self.image_details_frame, text="Prompt kopieren", command=self._copy_prompt_to_clipboard, width=120, height=28, corner_radius=8)
        self.copy_prompt_button.grid(row=0, column=1, padx=5, pady=0, sticky="e")
        self.copy_seed_button = ctk.CTkButton(self.image_details_frame, text="Seed kopieren", command=self._copy_seed_to_clipboard, width=120, height=28, corner_radius=8)
        self.copy_seed_button.grid(row=1, column=1, padx=5, pady=0, sticky="e")
        
        self.current_image_seed = -1 # Speichert den Seed des aktuellen Bildes


        # --- Buttons für Speichern und Galerie ---
        self.action_buttons_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.action_buttons_frame.grid(row=2, column=0, padx=20, pady=(10, 20), sticky="s") # Angepasste Zeile
        self.action_buttons_frame.grid_columnconfigure(0, weight=1)
        self.action_buttons_frame.grid_columnconfigure(1, weight=1)

        # Der "Speichern unter..." Button wird nun zum "Bild speichern" Button
        self.save_button = ctk.CTkButton(self.action_buttons_frame, text="Bild speichern", command=self.save_current_image_to_default_folder, state="disabled", corner_radius=8)
        self.save_button.grid(row=0, column=0, padx=5, pady=5)

        self.gallery_button = ctk.CTkButton(self.action_buttons_frame, text="Galerie öffnen", command=self.open_gallery, corner_radius=8)
        self.gallery_button.grid(row=0, column=1, padx=5, pady=5)


        # --- Statusleiste mit Ladebalken (am unteren Rand des Hauptfensters) ---
        self.status_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.status_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="ew") # Spannt über beide Spalten
        self.status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(self.status_frame, text=self.initial_status_message, text_color="gray", font=ctk.CTkFont(size=14)) # Initialisiere mit dynamischer Nachricht
        self.status_label.grid(row=0, column=0, padx=0, pady=0, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.status_frame, orientation="horizontal")
        self.progress_bar.grid(row=1, column=0, padx=0, pady=(5, 0), sticky="ew")
        self.progress_bar.set(0) # Start bei 0
        self.progress_bar.configure(mode="determinate") # Determinate für Prozent, Indeterminate für "Laden..."

        self.progress_percentage_label = ctk.CTkLabel(self.status_frame, text="", font=ctk.CTkFont(size=12))
        self.progress_percentage_label.grid(row=2, column=0, padx=0, pady=(0, 0), sticky="ew")


        self.loading_animation_id = None # Für die Ladeanimation
        self.current_generated_image = None # Speichert das PIL-Image des zuletzt generierten Bildes
        self.current_generated_prompt = None # Speichert den Prompt des zuletzt generierten Bildes
        self.current_generated_negative_prompt = None # Speichert den negativen Prompt
        self.pipe = None # Das geladene Stable Diffusion Pipeline-Objekt
        self.generation_thread = None # Referenz auf den Generierungs-Thread

        # Initialisiere das Galerie-Fenster als None
        self.gallery_window_instance = None

        # Rufen Sie _populate_model_list HIER auf, nachdem alle Widgets initialisiert wurden
        self._populate_model_list()

    def on_closing(self):
        """Wird aufgerufen, wenn das Fenster geschlossen wird."""
        if self.generation_thread and self.generation_thread.is_alive():
            self.stop_event.set() # Signalisiert dem Thread, dass er anhalten soll
            self.update_status("Generierung wird abgebrochen...", "orange")
            self.progress_bar.set(0)
            self.progress_percentage_label.configure(text="Abbruch...")
            # Geben Sie dem Thread kurz Zeit, sich zu beenden
            # In der Realität ist ein pipe-Aufruf schwer zu unterbrechen, aber das Signal ist gesetzt
            self.generation_thread.join(timeout=2) 
        self.destroy() # Zerstört das Fenster

    def update_status(self, message, color="gray"):
        """Aktualisiert die Statusleiste."""
        self.status_label.configure(text=message, text_color=color)

    def _update_progress_bar(self, value, text=""):
        """Aktualisiert den Ladebalken und das Prozent-Label."""
        self.progress_bar.set(value)
        self.progress_percentage_label.configure(text=text)
        self.update_idletasks() # Stellt sicher, dass die GUI aktualisiert wird

    def start_loading_animation(self, base_message="Generiere Bild", mode="indeterminate"):
        """Startet eine Ladeanimation (determinate oder indeterminate)."""
        self.progress_bar.configure(mode=mode)
        if mode == "indeterminate":
            self.progress_bar.start()
            self.progress_percentage_label.configure(text="") # Keine Prozent bei indeterminate
            dots = 0
            def animate():
                nonlocal dots
                dots = (dots + 1) % 4
                self.status_label.configure(text=base_message + "." * dots, text_color="blue")
                if self.loading_animation_id:
                    self.loading_animation_id = self.after(300, animate)
            self.loading_animation_id = self.after(0, animate)
        else: # determinate
            self.progress_bar.stop()
            self.progress_bar.set(0)
            self.progress_percentage_label.configure(text="0%")
            self.status_label.configure(text=base_message, text_color="blue")


    def stop_loading_animation(self):
        """Stoppt die Ladeanimation."""
        if self.loading_animation_id:
            self.after_cancel(self.loading_animation_id)
            self.loading_animation_id = None
        self.progress_bar.stop()
        self.progress_bar.set(0)
        self.progress_percentage_label.configure(text="")


    def _update_steps_label(self, value):
        """Aktualisiert das Label für die Steps."""
        self.steps_value_label.configure(text=f"{int(value)}")

    def _update_cfg_label(self, value):
        """Aktualisiert das Label für die CFG-Skala."""
        self.cfg_value_label.configure(text=f"{value:.1f}")

    def _set_random_seed(self):
        """Setzt einen zufälligen Seed im Eingabefeld."""
        self.seed_entry.delete(0, ctk.END)
        self.seed_entry.insert(0, str(random.randint(0, 2**32 - 1))) # Zufällige 32-Bit Ganzzahl

    def _update_size_options(self, value):
        """Callback für Größen-Optionen (falls Logik für Abhängigkeiten nötig wäre)."""
        # Aktuell keine spezielle Logik nötig, aber der Callback muss existieren.
        pass

    def _set_settings_state(self, state):
        """Setzt den Zustand der Einstellungswidgets (normal/disabled)."""
        self.width_optionmenu.configure(state=state)
        self.height_optionmenu.configure(state=state)
        self.steps_slider.configure(state=state)
        self.seed_entry.configure(state=state)
        self.random_seed_button.configure(state=state)
        self.cfg_slider.configure(state=state)
        self.scheduler_optionmenu.configure(state=state)
        self.prompt_entry.configure(state=state) # Prompt-Feld
        self.negative_prompt_entry.configure(state=state) # Negativer Prompt-Feld
        self.clear_prompt_button.configure(state=state) # Clear-Button
        self.custom_width_entry.configure(state=state) # Eigene Größe
        self.custom_height_entry.configure(state=state) # Eigene Größe
        self.use_custom_size_checkbox.configure(state=state) # Eigene Größe Checkbox
        self.num_images_entry.configure(state=state) # Anzahl Bilder
        # self.live_preview_checkbox.configure(state=state) # Live-Vorschau Checkbox (entfernt)
        # 8-Bit Checkbox bleibt aktiv, wenn GPU verfügbar ist, da sie das Laden beeinflusst
        if torch.cuda.is_available() and not self.force_cpu: # Nur aktivieren, wenn GPU verfügbar und nicht CPU-Modus
            self.quantization_checkbox.configure(state="normal" if state == "normal" else "disabled")
        else:
            self.quantization_checkbox.configure(state="disabled") # Immer deaktiviert, wenn CPU-Modus
        self.is_sdxl_checkbox.configure(state="normal" if state == "normal" else "disabled") # SDXL-Checkbox auch steuern


    def _toggle_quantization_info(self):
        """Zeigt/versteckt zusätzliche Info zur Quantisierung."""
        # Diese Methode ist nur ein Platzhalter, falls Sie visuelles Feedback wünschen.
        # Die eigentliche Logik wird beim Laden des Modells ausgeführt.
        pass

    def _toggle_custom_size(self):
        """Aktiviert/Deaktiviert die Eingabefelder für eigene Größe."""
        if self.use_custom_size_checkbox.get():
            self.width_optionmenu.configure(state="disabled")
            self.height_optionmenu.configure(state="disabled")
            self.custom_width_entry.configure(state="normal")
            self.custom_height_entry.configure(state="normal")
        else:
            self.width_optionmenu.configure(state="normal")
            self.height_optionmenu.configure(state="normal")
            self.custom_width_entry.configure(state="disabled")
            self.custom_height_entry.configure(state="disabled")

    def _clear_prompt(self):
        """Löscht den Inhalt des Prompt-Eingabefeldes."""
        self.prompt_entry.delete(0, ctk.END)
        self.negative_prompt_entry.delete(0, ctk.END)

    def _load_prompt_history(self):
        """Lädt den Prompt-Verlauf aus einer Datei."""
        if os.path.exists(PROMPT_HISTORY_FILE):
            try:
                with open(PROMPT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_list = json.load(f)
                    self.prompt_history.extend(history_list)
            except json.JSONDecodeError:
                pass # Datei ist leer oder korrupt
        self._update_prompt_history_options()

    def _save_prompt_history(self):
        """Speichert den aktuellen Prompt-Verlauf in einer Datei."""
        with open(PROMPT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(list(self.prompt_history), f, indent=4, ensure_ascii=False)

    def _add_to_prompt_history(self, prompt, negative_prompt):
        """Fügt einen Prompt zum Verlauf hinzu."""
        entry = {"prompt": prompt, "negative_prompt": negative_prompt}
        if entry not in self.prompt_history: # Vermeide Duplikate
            self.prompt_history.appendleft(entry) # Fügt am Anfang hinzu
            self._update_prompt_history_options()
            self._save_prompt_history()

    def _update_prompt_history_options(self):
        """Aktualisiert die Optionen im Prompt-Verlauf Dropdown-Menü."""
        if not self.prompt_history:
            self.prompt_history_optionmenu.configure(values=["Kein Verlauf"])
            self.prompt_history_optionmenu.set("Kein Verlauf")
            self.prompt_history_optionmenu.configure(state="disabled")
        else:
            # Zeige nur den positiven Prompt im Menü an
            options = [entry["prompt"] for entry in self.prompt_history]
            self.prompt_history_optionmenu.configure(values=options)
            self.prompt_history_optionmenu.set(options[0]) # Setze den neuesten als Standard
            self.prompt_history_optionmenu.configure(state="normal")

    def _load_prompt_from_history(self, selected_prompt_text):
        """Lädt einen ausgewählten Prompt aus dem Verlauf in die Eingabefelder."""
        for entry in self.prompt_history:
            if entry["prompt"] == selected_prompt_text:
                self.prompt_entry.delete(0, ctk.END)
                self.prompt_entry.insert(0, entry["prompt"])
                self.negative_prompt_entry.delete(0, ctk.END)
                self.negative_prompt_entry.insert(0, entry.get("negative_prompt", ""))
                self.update_status(f"Prompt aus Verlauf geladen: '{selected_prompt_text}'", "gray")
                break


    def _copy_to_clipboard(self, text):
        """Kopiert Text in die Zwischenablage."""
        if pyperclip:
            try:
                pyperclip.copy(text)
                self.update_status("In Zwischenablage kopiert!", "green")
            except pyperclip.PyperclipException:
                self.update_status("Fehler beim Kopieren in Zwischenablage (pyperclip Problem).", "red")
        else:
            # Fallback für Tkinter-Zwischenablage, weniger robust
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update_status("In Zwischenablage kopiert (Fallback)!", "green")

    def _copy_prompt_to_clipboard(self):
        """Kopiert den aktuellen Prompt in die Zwischenablage."""
        if self.current_generated_prompt:
            self._copy_to_clipboard(self.current_generated_prompt)
        else:
            self.update_status("Kein Prompt zum Kopieren vorhanden.", "orange")

    def _copy_seed_to_clipboard(self):
        """Kopiert den aktuellen Seed in die Zwischenablage."""
        if self.current_image_seed != -1:
            self._copy_to_clipboard(str(self.current_image_seed))
        else:
            self.update_status("Kein Seed zum Kopieren vorhanden.", "orange")


    def _detect_sdxl_model(self, model_path):
        """
        Versucht, anhand der Safetensors-Datei zu erkennen, ob es sich um ein SDXL-Modell handelt.
        Dies geschieht durch Überprüfung auf Metadaten, SDXL-spezifische Schlüssel im State-Dictionary
        und Dateigröße.
        """
        if not model_path or not os.path.exists(model_path):
            print(f"DEBUG: Modellpfad existiert nicht für SDXL-Erkennung: {model_path}")
            return False

        is_sdxl_detected = False
        
        # 1. Prüfe den Dateinamen
        filename = os.path.basename(model_path).lower()
        if "sdxl" in filename or "flux" in filename: # Auch "flux" als Indikator hinzufügen
            is_sdxl_detected = True
            print(f"DEBUG: SDXL/Flux erkannt über Dateinamen: '{filename}'")
            return is_sdxl_detected # Sofort zurückgeben, wenn im Dateinamen gefunden

        # 2. Prüfe die Dateigröße (größer als 6 GB ist ein starker Indikator für SDXL)
        try:
            file_size_bytes = os.path.getsize(model_path)
            file_size_gb = file_size_bytes / (1024**3)
            print(f"DEBUG: Dateigröße von '{filename}': {file_size_gb:.2f} GB")
            if file_size_gb > 6.0: # Schwellenwert von 6 GB
                is_sdxl_detected = True
                print(f"DEBUG: SDXL erkannt über Dateigröße (>6GB).")
                return is_sdxl_detected # Sofort zurückgeben, wenn Größe passt
        except Exception as e:
            print(f"FEHLER: Konnte Dateigröße für SDXL-Erkennung nicht ermitteln: {e}")
            # Fahren Sie fort, um andere Erkennungsmethoden zu versuchen

        try:
            with safetensors.torch.safe_open(model_path, framework="pt") as f:
                # 3. Prüfe auf Metadaten
                metadata = f.metadata()
                print(f"DEBUG: Metadaten im Safetensors-Header: {metadata}")
                if metadata and "model_type" in metadata and "stable-diffusion-xl" in metadata["model_type"].lower():
                    is_sdxl_detected = True
                    print(f"DEBUG: SDXL erkannt über Metadaten: {metadata.get('model_type')}")
                    return is_sdxl_detected # Sofort zurückgeben, wenn in Metadaten gefunden

                # 4. Prüfe auf spezifische Schlüssel (umfassender)
                # Lese nur eine begrenzte Anzahl von Schlüsseln, um Performance zu sparen,
                # da SDXL-Indikatoren typischerweise am Anfang der Schlüsselstruktur stehen.
                # Wenn das Modell sehr viele Schlüssel hat, könnte das Lesen aller Schlüssel langsam sein.
                # Eine gute Balance ist hier wichtig.
                keys_to_check_limit = 500 # Erhöhtes Limit, falls relevante Schlüssel weiter hinten liegen
                all_keys = list(f.keys())
                print(f"DEBUG: Überprüfe bis zu {keys_to_check_limit} Schlüssel von insgesamt {len(all_keys)} für SDXL-Indikatoren.")

                sdxl_key_indicators = [
                    "text_encoder_2", # Zweiter Text-Encoder
                    "tokenizer_2",    # Zweiter Tokenizer
                    "text_model_2",   # Alternativer Name für zweiten Text-Encoder
                    "conditioner.embedders.1.transformer.text_model", # Pfad zum zweiten Text-Encoder in einigen Formaten
                    "model.text_encoder_2.text_model.embeddings.token_embedding.weight", # Spezifischer Gewichtsschlüssel
                    "model.clip_l.transformer.resblocks.11.attn.q_proj.weight", # Spezifisch für SDXLs zweiter Text-Encoder-Pfad
                    "model.clip_g.transformer.resblocks.23.attn.q_proj.weight" # Spezifisch für SDXLs erster Text-Encoder-Pfad (falls vorhanden)
                ]

                for key in all_keys[:keys_to_check_limit]: # Prüfe nur die ersten X Schlüssel
                    for indicator in sdxl_key_indicators:
                        if indicator in key:
                            is_sdxl_detected = True
                            print(f"DEBUG: SDXL-Indikator '{indicator}' gefunden in Schlüssel: {key}")
                            break # Sobald ein Indikator gefunden, reicht es
                    if is_sdxl_detected:
                        break

                print(f"DEBUG: Ergebnis der SDXL-Schlüsselprüfung: {is_sdxl_detected}")
                return is_sdxl_detected

        except Exception as e:
            print(f"FEHLER: Fehler beim automatischen Erkennen des Modelltyps: {e}")
            traceback.print_exc()
            return False

    def _populate_model_list(self):
        """Füllt das Modell-Dropdown-Menü mit .safetensors-Dateien aus dem MODELS_DIR."""
        model_files = []
        try:
            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR) # Stelle sicher, dass der Ordner existiert
                print(f"DEBUG: Modelle-Ordner '{MODELS_DIR}' wurde erstellt.")

            for filename in os.listdir(MODELS_DIR):
                if filename.endswith(".safetensors"):
                    model_files.append(os.path.splitext(filename)[0]) # Füge nur den Namen ohne Endung hinzu
            
            if not model_files:
                self.model_optionmenu.configure(values=["Keine Modelle gefunden"])
                self.model_optionmenu.set("Keine Modelle gefunden")
                self.model_optionmenu.configure(state="disabled")
                self.update_status(f"Keine .safetensors-Modelle im '{MODELS_DIR}' Ordner gefunden.", "orange")
            else:
                self.model_optionmenu.configure(values=model_files)
                self.model_optionmenu.set(model_files[0]) # Wähle das erste Modell standardmäßig aus
                self.model_optionmenu.configure(state="normal")
                self._on_model_select(model_files[0]) # Automatische Erkennung für das erste Modell ausführen
                self.update_status(f"{len(model_files)} Modelle im '{MODELS_DIR}' Ordner gefunden.", "gray")

        except Exception as e:
            self.model_optionmenu.configure(values=["Fehler beim Laden von Modellen"])
            self.model_optionmenu.set("Fehler beim Laden von Modellen")
            self.model_optionmenu.configure(state="disabled")
            self.update_status(f"Fehler beim Laden der Modelle aus '{MODELS_DIR}': {e}", "red")
            traceback.print_exc()


    def _on_model_select(self, selected_model_name):
        """Wird aufgerufen, wenn ein Modell aus dem Dropdown ausgewählt wird."""
        if selected_model_name == "Keine Modelle gefunden" or not selected_model_name:
            return # Nichts tun, wenn keine Modelle ausgewählt werden können

        model_full_path = os.path.join(MODELS_DIR, selected_model_name + ".safetensors")
        
        # Automatische SDXL-Erkennung für die ausgewählte Datei
        is_sdxl_detected = self._detect_sdxl_model(model_full_path)
        if is_sdxl_detected:
            self.is_sdxl_checkbox.select()
            self.width_optionmenu.set("1024")
            self.height_optionmenu.set("1024")
            self.update_status(f"Modell ausgewählt: {selected_model_name}. SDXL-Modell erkannt. Standardauflösung auf 1024x1024 gesetzt.", "gray")
        else:
            self.is_sdxl_checkbox.deselect()
            self.width_optionmenu.set("512")
            self.height_optionmenu.set("512")
            self.update_status(f"Modell ausgewählt: {selected_model_name}. Standard SD-Modell erkannt. Standardauflösung auf 512x512 gesetzt.", "gray")


    def load_model(self):
        """Lädt das Stable Diffusion Modell in einem separaten Thread."""
        selected_display_name = self.model_optionmenu.get()
        if selected_display_name == "Keine Modelle gefunden" or not selected_display_name:
            self.update_status("Bitte zuerst ein Modell auswählen!", "orange")
            return
        
        model_path = os.path.join(MODELS_DIR, selected_display_name + ".safetensors")

        if not os.path.exists(model_path):
            self.update_status(f"Fehler: Modell '{selected_display_name}.safetensors' nicht gefunden unter diesem Pfad: {os.path.abspath(MODELS_DIR)}.", "red")
            return
        
        self.load_model_button.configure(state="disabled", text="Lade Modell...")
        self.model_optionmenu.configure(state="disabled") # Deaktiviere Modellauswahl während des Ladens
        self.prompt_entry.configure(state="disabled")
        self.negative_prompt_entry.configure(state="disabled") # Negativer Prompt deaktivieren
        self.clear_prompt_button.configure(state="disabled") # Clear-Button deaktivieren
        self.generate_button.configure(state="disabled")
        self._set_settings_state("disabled") # Deaktiviert alle Einstellungen während des Ladens

        self.image_label.configure(image=None, text="Lade Stable Diffusion Modell...\nDies kann einige Zeit dauhen und viel RAM/VRAM beanspruchen.", font=ctk.CTkFont(size=16), text_color="yellow")
        self.start_loading_animation(base_message="Lade Modell", mode="indeterminate")
        
        threading.Thread(target=self._load_model_thread, args=(model_path,)).start()

    def _load_model_thread(self, model_path):
        """Thread-Funktion zum Laden des Modells."""
        try:
            device = self.device # Verwende das in __init__ festgelegte Gerät
            self.update_status(f"Lade Modell auf {device.upper()}...", "blue")

            # --- Speicherbereinigung vor dem Laden eines neuen Modells ---
            if self.pipe is not None:
                print("DEBUG: Entlade vorheriges Modell aus dem Speicher...")
                del self.pipe
                self.pipe = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Leere GPU-Speicher
                gc.collect() # Python Garbage Collector aufrufen
                print("DEBUG: Vorheriges Modell entladen und Speicher bereinigt.")
            # --- Ende Speicherbereinigung ---

            # --- Diagnose GPU-Status (für Konsole) ---
            print("\n--- GPU Diagnose (Laden) ---")
            print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"Anzahl der CUDA-Geräte: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"Gerät {i}: {torch.cuda.get_device_name(i)}")
                print(f"Aktuelles CUDA-Gerät: {torch.cuda.current_device()}")
                print(f"PyTorch CUDA Version: {torch.version.cuda}")
            else:
                print("CUDA (NVIDIA GPU) ist nicht verfügbar. Überprüfen Sie Ihre Treiber und PyTorch-Installation.")
            print("-----------------------------\n")
            # --- Ende Diagnose ---

            load_in_8bit = self.quantization_checkbox.get() and device == "cuda"
            is_sdxl = self.is_sdxl_checkbox.get() # SDXL-Checkbox-Status abrufen

            if load_in_8bit:
                self.update_status("Lade Modell mit 8-Bit-Quantisierung...", "blue")
            
            # Wähle die richtige Pipeline-Klasse basierend auf der SDXL-Checkbox
            pipeline_class = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

            # Lade das Stable Diffusion Pipeline aus der safetensors-Datei
            self.pipe = pipeline_class.from_single_file(
                model_path,
                torch_dtype=torch.float16 if device == "cuda" and not load_in_8bit else torch.float32,
                low_cpu_mem_usage=True, # Hilft beim Laden großer Modelle in den Hauptspeicher
                load_in_8bit=load_in_8bit # Parameter für 8-Bit-Quantisierung
            )
            
            # --- Zusätzlicher Post-Load-Check für SDXL-Komponenten ---
            if is_sdxl and not hasattr(self.pipe, 'text_encoder_2'):
                self.after(0, self.update_status, "WARNUNG: SDXL-Modell geladen, aber 'text_encoder_2' nicht gefunden. Möglicherweise ist das Modell inkompatibel oder beschädigt. Generierung könnte fehlschlagen.", "orange")
                print("WARNING: SDXL model loaded, but text_encoder_2 not found. Generation might fail.")
            # --- Ende Post-Load-Check ---

            # --- Optimierungen anwenden ---
            if device == "cuda":
                # Aktiviere speichereffiziente Attention, falls xformers verfügbar ist
                try:
                    import xformers
                    self.pipe.enable_xformers_memory_efficient_attention()
                    self.update_status("xFormers aktiviert (falls verfügbar).", "blue")
                except ImportError:
                    self.update_status("xFormers nicht gefunden, Generierung ohne Speicheroptimierung.", "orange")
                
                # Kompiliere das Modell für schnellere Inferenz (PyTorch 2.0+)
                # Der try-except-block um torch.compile ist wichtig, da Triton-Installation fehlschlagen kann
                # und der Benutzer keine weiteren Downloads wünscht.
                # WICHTIG: torch.compile wird hier deaktiviert, da es den cl.exe Compiler benötigt.
                # try:
                #     self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
                #     self.update_status("Modell mit torch.compile optimiert (erste Generierung kann länger dauern).", "blue")
                # except Exception as compile_err:
                #     self.update_status(f"Fehler bei torch.compile: {compile_err}. Optimierung übersprungen.", "orange")
                #     traceback.print_exc() # Ausgabe des Fehlers in der Konsole
                
                # Aktiviere sequentielles CPU-Offloading, falls VRAM begrenzt ist
                # Dies macht die Generierung langsamer, ermöglicht aber größere Modelle
                # Nur aktivieren, wenn nicht bereits 8-Bit-Quantisierung verwendet wird, da sie sich überschneiden können
                if not load_in_8bit:
                    self.pipe.enable_model_cpu_offload()
                    self.update_status("Modell-CPU-Offloading aktiviert (Generierung wird langsamer, aber größere Modelle passen).", "orange")

                # VAE Slicing/Tiling für VRAM-Optimierung bei hohen Auflösungen
                try:
                    self.pipe.vae.enable_slicing()
                    self.pipe.vae.enable_tiling()
                    self.update_status("VAE Slicing/Tiling aktiviert für VRAM-Optimierung (hilft bei hohen Auflösungen).", "blue")
                except Exception as vae_err:
                    self.update_status(f"Fehler beim Aktivieren von VAE Slicing/Tiling: {vae_err}", "orange")

            # Modell auf das Gerät verschieben (relevant, wenn keine Offloading-Methode verwendet wird oder für CPU)
            # enable_model_cpu_offload() ruft intern .to("cpu") auf und verwaltet GPU-Transfers.
            # Daher ist ein explizites .to(device) hier nicht mehr nötig, wenn model_cpu_offload aktiv ist.
            # Wenn device == "cpu", muss es aber explizit auf CPU gesetzt werden.
            # Wenn load_in_8bit aktiv ist, wird das Modell bereits auf dem Gerät geladen.
            if device == "cpu" and not (hasattr(self.pipe, '_hf_accelerate_enabled') or load_in_8bit):
                self.pipe.to(device)
            # --- Ende Optimierungen ---

            self.after(0, self.stop_loading_animation)
            self.after(0, self.update_status, "Modell erfolgreich geladen!", "green")
            self.after(0, lambda: self.load_model_button.configure(state="normal", text="Modell laden"))
            self.after(0, lambda: self.model_optionmenu.configure(state="normal")) # Aktiviere Modellauswahl wieder
            self.after(0, lambda: self.prompt_entry.configure(state="normal"))
            self.after(0, lambda: self.negative_prompt_entry.configure(state="normal")) # Negativer Prompt aktivieren
            self.after(0, lambda: self.clear_prompt_button.configure(state="normal")) # Clear-Button aktivieren
            self.after(0, lambda: self.generate_button.configure(state="normal"))
            self.after(0, lambda: self._set_settings_state("normal")) # Aktiviert alle Einstellungen
            self.after(0, lambda: self.image_label.configure(text="Bereit zur Bildgenerierung."))

        except torch.cuda.OutOfMemoryError:
            self.after(0, self.stop_loading_animation)
            self.after(0, self.update_status, "Fehler: GPU-Speicher (VRAM) nicht ausreichend für dieses Modell.", "red")
            self.after(0, lambda: self.image_label.configure(text="Fehler: GPU-Speicher nicht ausreichend."))
            traceback.print_exc() # Ausgabe des Fehlers in der Konsole
            self._reset_ui_on_load_error()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.after(0, self.stop_loading_animation)
                self.after(0, self.update_status, f"Fehler: Speicher nicht ausreichend. Versuchen Sie ein kleineres Modell oder schließen Sie andere Anwendungen. ({e})", "red")
                self.after(0, lambda: self.image_label.configure(text="Fehler: Speicher nicht ausreichend."))
                traceback.print_exc() # Ausgabe des Fehlers in der Konsole
                self._reset_ui_on_load_error()
            else:
                self.after(0, self.stop_loading_animation)
                self.after(0, self.update_status, f"Fehler beim Laden des Modells: {e}", "red")
                self.after(0, lambda: self.image_label.configure(text="Fehler beim Laden des Modells. Überprüfen Sie den Pfad."))
                traceback.print_exc() # Ausgabe des Fehlers in der Konsole
                self._reset_ui_on_load_error()
        except Exception as e:
            self.after(0, self.stop_loading_animation)
            self.after(0, self.update_status, f"Fehler beim Laden des Modells: {e}", "red")
            self.after(0, lambda: self.image_label.configure(text="Fehler beim Laden des Modells. Bitte überprüfen Sie den Pfad und Ihre Installation."))
            traceback.print_exc() # Ausgabe des Fehlers in der Konsole
            self._reset_ui_on_load_error()
        finally:
            # Stelle sicher, dass GPU-Cache geleert wird, auch wenn ein Fehler auftritt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _reset_ui_on_load_error(self):
        """Setzt die UI-Elemente nach einem Ladefehler zurück."""
        self.pipe = None
        # Zusätzliche Speicherbereinigung bei Fehler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        self.load_model_button.configure(state="normal", text="Modell laden")
        self.model_optionmenu.configure(state="normal") # Aktiviere Modellauswahl wieder
        self.prompt_entry.configure(state="normal") # Prompt-Feld
        self.negative_prompt_entry.configure(state="normal") # Negativer Prompt deaktivieren
        self.clear_prompt_button.configure(state="normal") # Clear-Button aktivieren
        self.generate_button.configure(state="disabled")
        self._set_settings_state("normal") # Hier auf "normal" setzen, um Eingaben wieder zu ermöglichen
        # Die 8-Bit-Checkbox sollte ihren Zustand beibehalten, wenn die GPU verfügbar ist
        if torch.cuda.is_available():
            self.quantization_checkbox.configure(state="normal")
        self.is_sdxl_checkbox.configure(state="normal") # SDXL-Checkbox auch wieder aktivieren


    def generate_image_event(self, event=None):
        """Startet den Bildgenerierungsprozess in einem separaten Thread."""
        if not self.pipe:
            self.update_status("Bitte zuerst ein Modell laden!", "orange")
            return

        prompt = self.prompt_entry.get().strip()
        negative_prompt = self.negative_prompt_entry.get().strip() # Negativen Prompt auslesen

        if not prompt:
            self.update_status("Bitte eine Bildbeschreibung eingeben!", "orange")
            return

        # Einstellungen auslesen
        try:
            # Überprüfe, ob "Eigene Größe verwenden" aktiv ist
            if self.use_custom_size_checkbox.get():
                width = int(self.custom_width_entry.get())
                height = int(self.custom_height_entry.get())
                if width <= 0 or height <= 0:
                    raise ValueError("Breite und Höhe müssen positive Zahlen sein.")
            else:
                width = int(self.width_optionmenu.get())
                height = int(self.height_optionmenu.get())

            num_inference_steps = int(self.steps_slider.get())
            guidance_scale = float(self.cfg_slider.get())
            seed_str = self.seed_entry.get().strip()
            seed = int(seed_str) if seed_str and seed_str != "-1" else -1
            selected_scheduler_name = self.scheduler_optionmenu.get()
            num_images = int(self.num_images_entry.get()) # Anzahl der Bilder auslesen
            if num_images <= 0:
                raise ValueError("Anzahl der Bilder muss positiv sein.")

        except ValueError as e:
            self.update_status(f"Fehler in den Einstellungen: {e}. Bitte gültige Zahlen eingeben.", "red")
            return

        self.current_generated_prompt = prompt
        # Speichern des negativen Prompts für Metadaten, falls benötigt
        self.current_generated_negative_prompt = negative_prompt if negative_prompt else ""

        self.save_button.configure(state="disabled")
        self.generate_button.configure(state="disabled", text="Generiere...")
        self.prompt_entry.configure(state="disabled")
        self.negative_prompt_entry.configure(state="disabled") # Negativer Prompt deaktivieren
        self.clear_prompt_button.configure(state="disabled") # Clear-Button deaktivieren
        self._set_settings_state("disabled")

        self.image_label.configure(image=None, text="Generiere Bild...\nDies kann je nach Hardware einige Zeit dauern.", font=ctk.CTkFont(size=16), text_color="yellow")
        self.start_loading_animation(base_message="Generiere Bild", mode="determinate")

        # Erstelle Generator für den Seed
        # Der Generator muss auf dem richtigen Gerät sein
        generator = torch.Generator(device=self.pipe.device if hasattr(self.pipe, 'device') else "cpu") 
        if seed != -1:
            generator = generator.manual_seed(seed)
        else:
            random_seed = random.randint(0, 2**32 - 1)
            generator = generator.manual_seed(random_seed)
            self.seed_entry.delete(0, ctk.END)
            self.seed_entry.insert(0, str(random_seed))

        # Setze den ausgewählten Scheduler für die Pipeline
        if selected_scheduler_name in self.scheduler_map:
            # Hier müssen wir die Konfiguration des aktuellen Schedulers abrufen
            # und dann den neuen Scheduler mit dieser Konfiguration und ggf. Karras-Optionen erstellen.
            current_scheduler_config = self.pipe.scheduler.config
            new_scheduler_config = dict(current_scheduler_config) # Kopie erstellen

            if "Karras" in selected_scheduler_name:
                new_scheduler_config["use_karras_sigmas"] = True
            else:
                # Sicherstellen, dass use_karras_sigmas auf False gesetzt ist, wenn es keine Karras-Variante ist
                if "use_karras_sigmas" in new_scheduler_config:
                    new_scheduler_config["use_karras_sigmas"] = False

            self.pipe.scheduler = self.scheduler_map[selected_scheduler_name].from_config(new_scheduler_config)
        else:
            self.update_status(f"Unbekannter Scheduler: {selected_scheduler_name}. Verwende Standard-Scheduler.", "orange")

        self.generation_thread = threading.Thread(target=self._generate_images_thread_loop, 
                                                  args=(prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, generator, num_images))
        self.generation_thread.start()

    def _progress_callback(self, pipeline_instance, step, timestep, callback_kwargs): # Angepasste Signatur
        """Callback-Funktion für den Fortschritt der Bildgenerierung."""
        total_steps_per_image = int(self.steps_slider.get())
        
        # Extract context from callback_kwargs
        current_image_index = callback_kwargs.get("current_image_index", 0)
        total_images = callback_kwargs.get("total_images", 1)

        # Calculate progress for the current image (0.0 to 1.0)
        progress_within_current_image = step / total_steps_per_image if total_steps_per_image > 0 else 0

        # Calculate overall progress (0.0 to 1.0 across all images)
        # Each image represents a fraction of the total progress
        overall_progress_value = (current_image_index + progress_within_current_image) / total_images
        overall_percentage = int(overall_progress_value * 100)

        self.after(0, self._update_progress_bar, overall_progress_value, f"{overall_percentage}%")
        self.after(0, self.update_status, f"Generiere Bild {current_image_index+1}/{total_images}... Schritt {step}/{total_steps_per_image}", "blue")

        # Live-Vorschau ist entfernt worden, daher wird dieser Block nicht mehr ausgeführt
        # if self.live_preview_checkbox.get():
        #     latents = callback_kwargs['latents']
        #     
        #     if latents.device != self.pipe.vae.device:
        #         latents = latents.to(self.pipe.vae.device)
        #
        #     latents = 1 / self.pipe.vae.config.scaling_factor * latents
        #     with torch.no_grad():
        #         decoded_image_tensor = self.pipe.vae.decode(latents).sample
        #
        #     decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)
        #     decoded_image_numpy = decoded_image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        #     pil_image = Image.fromarray((decoded_image_numpy * 255).astype(np.uint8))
        #     
        #     self.after(0, self._display_generated_image_live, pil_image)


        if self.stop_event.is_set():
            raise StopIteration
        
        return callback_kwargs # Wichtig: Rückgabe von callback_kwargs


    def _generate_images_thread_loop(self, prompt, negative_prompt, width, height, num_inference_steps, guidance_scale, generator, num_images):
        """Schleife für die Generierung mehrerer Bilder."""
        # generated_images_data wird hier nicht mehr benötigt, da Bilder direkt gespeichert werden
        # generated_images_data = [] 

        for i in range(num_images):
            if self.stop_event.is_set():
                self.after(0, self.update_status, f"Generierung von Bild {i+1}/{num_images} abgebrochen.", "orange")
                break

            self.after(0, lambda idx=i, total=num_images: self.image_label.configure(text=f"Generiere Bild {idx+1} von {total}...")) # Zeigt im Bildbereich an, Korrektur des lambda
            self.after(0, self._update_progress_bar, (i / num_images), f"{int((i / num_images) * 100)}%") # Setze Fortschritt für jedes neue Bild zurück auf den Beginn des aktuellen Bildes
            self.after(0, lambda: self.image_label.configure(image=None)) # Leere das Bildfeld vor neuer Generierung

            current_seed = generator.initial_seed() # Den aktuellen Seed für dieses Bild abrufen
            current_generator = torch.Generator(device=generator.device).manual_seed(current_seed) # Neuen Generator mit diesem Seed erstellen

            start_time = time.time() # Startzeit für Generierungsdauer

            try:
                pipeline_output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None, # Übergebe None, wenn leer
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=current_generator, # Verwende den spezifischen Generator
                    callback_on_step_end=self._progress_callback,
                    callback_on_step_end_user_data={"callback_steps": 1, "current_image_index": i, "total_images": num_images} # Pass context data
                )
                
                end_time = time.time() # Endzeit für Generierungsdauer
                generation_duration = end_time - start_time

                if pipeline_output and hasattr(pipeline_output, 'images') and \
                   isinstance(pipeline_output.images, list) and len(pipeline_output.images) > 0 and \
                   isinstance(pipeline_output.images[0], Image.Image): # Überprüfe, ob es ein PIL-Bild ist
                    
                    image = pipeline_output.images[0]
                    self.current_generated_image = image # Speichert das PIL-Image
                    self.current_image_seed = current_seed # Speichere den tatsächlichen Seed
                    self.after(0, self._display_generated_image, image) # Zeige das finale Bild an
                    self.after(0, self.update_status, f"Bild {i+1}/{num_images} erfolgreich generiert!", "green")
                    # Der Fortschrittsbalken wird bereits durch den Callback auf den korrekten Wert gesetzt.
                    # Dies stellt sicher, dass er am Ende des Bildes auf den korrekten Gesamtprozentsatz springt.
                    # self.after(0, self._update_progress_bar, 1.0, "100%") # DIESE ZEILE ENTFERNEN

                    # Automatisch das Bild speichern
                    self.after(0, self.save_current_image_to_default_folder) # Rufe die automatische Speicherfunktion auf
                    
                    # Aktualisiere die Details unter dem Bild
                    self.after(0, lambda: self.details_prompt_label.configure(text=f"Prompt: {prompt}"))
                    self.after(0, lambda: self.details_negative_prompt_label.configure(text=f"Negativ: {negative_prompt if negative_prompt else 'Kein negativer Prompt'}"))
                    self.after(0, lambda: self.details_params_label.configure(text=f"Größe: {width}x{height} | Schritte: {num_inference_steps} | CFG: {guidance_scale:.1f} | Seed: {self.current_image_seed} | Scheduler: {self.scheduler_optionmenu.get()}"))
                    self.after(0, lambda: self.details_generation_time_label.configure(text=f"Dauer: {generation_duration:.2f} Sekunden")) # Anzeige der Dauer

                    # Füge den Prompt zum Verlauf hinzu
                    self.after(0, self._add_to_prompt_history, prompt, negative_prompt)

                else:
                    self.after(0, self.update_status, f"Fehler bei Bild {i+1}/{num_images}: Keine gültigen Bilder von der Pipeline erhalten. Speicher oder Modell inkompatibel.", "red")
                    self.after(0, lambda: self.image_label.configure(text=f"Fehler bei Bild {i+1}/{num_images} (keine Bilder)."))
                    self.current_generated_image = None
                    self.current_image_seed = -1
                    self.after(0, lambda: self.details_prompt_label.configure(text="Prompt: "))
                    self.after(0, lambda: self.details_negative_prompt_label.configure(text="Negativ: "))
                    self.after(0, lambda: self.details_params_label.configure(text="Parameter: "))
                    self.after(0, lambda: self.details_generation_time_label.configure(text="Dauer: N/A")) # Dauer bei Fehler N/A

            except StopIteration:
                self.after(0, self.update_status, f"Generierung von Bild {i+1}/{num_images} abgebrochen.", "orange")
                self.after(0, lambda: self.image_label.configure(text=f"Generierung von Bild {i+1}/{num_images} abgebrochen."))
                self.current_generated_image = None
                self.current_image_seed = -1
                self.after(0, lambda: self.details_prompt_label.configure(text="Prompt: "))
                self.after(0, lambda: self.details_negative_prompt_label.configure(text="Negativ: "))
                self.after(0, lambda: self.details_params_label.configure(text="Parameter: "))
                self.after(0, lambda: self.details_generation_time_label.configure(text="Dauer: Abgebrochen")) # Dauer bei Abbruch
                break # Abbruch der Schleife bei StopIteration
            except torch.cuda.OutOfMemoryError:
                self.after(0, self.update_status, f"Fehler bei Bild {i+1}/{num_images}: GPU-Speicher (VRAM) nicht ausreichend. Versuchen Sie kleinere Bildgrößen oder weniger Schritte.", "red")
                self.after(0, lambda: self.image_label.configure(text=f"Fehler bei Bild {i+1}/{num_images}: GPU-Speicher nicht ausreichend."))
                traceback.print_exc()
                self.current_generated_image = None
                self.current_image_seed = -1
                self.after(0, lambda: self.details_prompt_label.configure(text="Prompt: "))
                self.after(0, lambda: self.details_negative_prompt_label.configure(text="Negativ: "))
                self.after(0, lambda: self.details_params_label.configure(text="Parameter: "))
                self.after(0, lambda: self.details_generation_time_label.configure(text="Dauer: Fehler")) # Dauer bei Fehler
                break # Abbruch der Schleife bei OOM-Fehler
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    self.after(0, self.update_status, f"Fehler bei Bild {i+1}/{num_images}: Speicher nicht ausreichend. Versuchen Sie kleinere Bildgrößen oder weniger Schritte. ({e})", "red")
                    self.after(0, lambda: self.image_label.configure(text=f"Fehler bei Bild {i+1}/{num_images}: Speicher nicht ausreichend."))
                else:
                    self.after(0, self.update_status, f"Fehler bei Bild {i+1}/{num_images}: {e}", "red")
                    self.after(0, lambda: self.image_label.configure(text=f"Fehler bei Bild {i+1}/{num_images}."))
                traceback.print_exc()
                self.current_generated_image = None
                self.current_image_seed = -1
                self.after(0, lambda: self.details_prompt_label.configure(text="Prompt: "))
                self.after(0, lambda: self.details_negative_prompt_label.configure(text="Negativ: "))
                self.after(0, lambda: self.details_params_label.configure(text="Parameter: "))
                self.after(0, lambda: self.details_generation_time_label.configure(text="Dauer: Fehler")) # Dauer bei Fehler
                break # Abbruch der Schleife bei RuntimeError
            except Exception as e:
                self.after(0, self.update_status, f"Fehler bei Bild {i+1}/{num_images}: {e}", "red")
                self.after(0, lambda: self.image_label.configure(text=f"Fehler bei Bild {i+1}/{num_images}."))
                traceback.print_exc()
                self.current_generated_image = None
                self.current_image_seed = -1
                self.after(0, lambda: self.details_prompt_label.configure(text="Prompt: "))
                self.after(0, lambda: self.details_negative_prompt_label.configure(text="Negativ: "))
                self.after(0, lambda: self.details_params_label.configure(text="Parameter: "))
                self.after(0, lambda: self.details_generation_time_label.configure(text="Dauer: Fehler")) # Dauer bei Fehler
                break # Abbruch der Schleife bei anderen Fehlern
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() # Leere GPU-Speicher nach jeder Generierung

        self.after(0, self._reset_ui_after_generation)
        # Nachdem alle Bilder generiert wurden, aktualisiere die Galerie (falls geöffnet)
        # Sicherstellen, dass der Fortschrittsbalken am Ende wirklich 100% ist, wenn alle Bilder erfolgreich waren
        if not self.stop_event.is_set():
            self.after(0, self._update_progress_bar, 1.0, "100% (Fertig)")
        self.after(0, self._update_gallery_if_open)

    def _reset_ui_after_generation(self):
        """Setzt die UI-Elemente nach der Generierung oder einem Abbruch zurück."""
        self.after(0, self.stop_loading_animation)
        self.after(0, lambda: self.generate_button.configure(state="normal", text="Bild generieren")) 
        self.after(0, lambda: self.prompt_entry.configure(state="normal")) 
        self.after(0, lambda: self.negative_prompt_entry.configure(state="normal")) # Negativer Prompt aktivieren
        self.after(0, lambda: self.clear_prompt_button.configure(state="normal")) # Clear-Button aktivieren
        self.after(0, lambda: self._set_settings_state("normal"))
        # Der Save-Button sollte nur aktiviert sein, wenn ein Bild da ist
        if self.current_generated_image:
            self.after(0, lambda: self.save_button.configure(state="normal"))
        else:
            self.after(0, lambda: self.save_button.configure(state="disabled"))


    def _display_generated_image_live(self, pil_image):
        """Zeigt ein Bild für die Live-Vorschau an."""
        # Dieser Code-Block ist jetzt deaktiviert, da Live-Vorschau entfernt wurde.
        pass

    def _display_generated_image(self, pil_image):
        """Zeigt das finale generierte PIL-Bild in der GUI an."""
        try:
            self.update_idletasks()
            # Der image_label ist jetzt im right_panel, das sich ausdehnt.
            # Wir müssen die Größe des right_panel.winfo_width/height verwenden.
            display_width = self.right_panel.winfo_width() - 40 # Polsterung von 20px auf jeder Seite
            display_height = self.right_panel.winfo_height() - self.image_details_frame.winfo_height() - self.action_buttons_frame.winfo_height() - 60 # Platz für Details, Buttons und Polsterung

            if display_width <= 0 or display_height <= 0:
                # Fallback-Werte, falls winfo_width/height noch nicht korrekt sind
                display_width = 700
                display_height = 500

            img_width, img_height = pil_image.size
            aspect_ratio = img_width / img_height

            if display_width / display_height > aspect_ratio:
                new_height = display_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = display_width
                new_height = int(new_width / aspect_ratio)

            image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Hier wird CTkImage verwendet
            ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(new_width, new_height))

            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image # Referenz speichern
            self.update_status("Bild erfolgreich generiert!", "green")
            self.current_generated_image = pil_image
        except Exception as e:
            self.update_status(f"Fehler beim Anzeigen des Bildes: {e}", "red")
            self.image_label.configure(image=None, text="Fehler beim Laden des Bildes.")
            self.current_generated_image = None
            self.save_button.configure(state="disabled")

    def save_current_image_to_default_folder(self):
        """
        Speichert das aktuell angezeigte Bild automatisch im Standardordner
        und aktualisiert die Metadaten.
        """
        if not self.current_generated_image or not self.current_generated_prompt:
            self.update_status("Kein Bild zum Speichern vorhanden.", "orange")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_{timestamp}.png"
            filepath = os.path.join(IMAGE_DIR, filename)

            self.current_generated_image.save(filepath)

            # Metadaten laden, aktualisieren und speichern
            metadata = {}
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    try:
                        metadata = json.load(f)
                    except json.JSONDecodeError:
                        metadata = {}

            metadata[filename] = {
                "prompt": self.current_generated_prompt,
                "negative_prompt": self.current_generated_negative_prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Zeitstempel im lesbaren Format
                "filepath": filepath # Speichere den vollständigen Pf
            }

            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            self.update_status(f"Bild automatisch gespeichert: {filename}", "green")
            # Der save_button wird hier nicht deaktiviert, da er jetzt "Speichern unter..." ist
            # und das automatische Speichern eine separate Funktion ist.
            # self.save_button.configure(state="disabled") # Diese Zeile wurde entfernt
        except Exception as e:
            self.update_status(f"Fehler beim automatischen Speichern des Bildes: {e}", "red")
            traceback.print_exc() # Ausgabe des Fehlers in der Konsole


    def open_gallery(self):
        """Öffnet ein neues Fenster, um die gespeicherten Bilder anzuzeigen."""
        # Überprüfen, ob bereits eine Galerie offen ist
        if self.gallery_window_instance and self.gallery_window_instance.winfo_exists():
            self.gallery_window_instance.focus_set() # Fokus auf bestehendes Fenster
            self.update_status("Galerie ist bereits geöffnet.", "orange")
            return

        gallery_window = ctk.CTkToplevel(self)
        gallery_window.title("Bild-Galerie")
        gallery_window.geometry("800x600")
        gallery_window.transient(self)
        gallery_window.protocol("WM_DELETE_WINDOW", lambda: self._on_gallery_close(gallery_window)) # Callback beim Schließen

        gallery_window.grid_columnconfigure(0, weight=1)
        gallery_window.grid_rowconfigure(0, weight=1)

        scrollable_frame = ctk.CTkScrollableFrame(gallery_window)
        scrollable_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        scrollable_frame.grid_columnconfigure(0, weight=1)

        self.gallery_scrollable_frame = scrollable_frame # Speichere Referenz für Updates
        self.gallery_window_instance = gallery_window # Speichere Instanz des Galerie-Fensters

        self._load_gallery_images() # Lade die Bilder in die Galerie

        # Button zum Löschen aller Bilder in der Galerie
        clear_all_button = ctk.CTkButton(gallery_window, text="Alle Bilder löschen", command=lambda: self._confirm_clear_all_images(gallery_window))
        clear_all_button.grid(row=1, column=0, pady=10)

    def _on_gallery_close(self, gallery_window):
        """Wird aufgerufen, wenn das Galerie-Fenster geschlossen wird."""
        self.gallery_window_instance = None # Setze die Instanz zurück
        gallery_window.destroy()

    def _update_gallery_if_open(self):
        """Aktualisiert die Galerie, falls sie geöffnet ist."""
        if self.gallery_window_instance and self.gallery_window_instance.winfo_exists():
            # Entferne alle alten Bilder aus dem ScrollableFrame
            for widget in self.gallery_scrollable_frame.winfo_children():
                widget.destroy()
            self._load_gallery_images() # Lade die aktualisierten Bilder

    def _load_gallery_images(self):
        """Lädt die Bilder in das Galerie-ScrollableFrame."""
        # Metadaten laden
        images_data = {}
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                try:
                    images_data = json.load(f)
                    images_data = {k: v for k, v in images_data.items() if "prompt" in v and "timestamp" in v and "filepath" in v}
                except json.JSONDecodeError:
                    pass

        if not images_data:
            ctk.CTkLabel(self.gallery_scrollable_frame, text="Noch keine Bilder gespeichert.", font=ctk.CTkFont(size=16), text_color="gray").pack(pady=20)
            return

        # Sortiere Bilder nach Zeitstempel (neuestes zuerst)
        sorted_filenames = sorted(images_data.keys(), key=lambda k: images_data[k].get("timestamp", ""), reverse=True)

        for filename in sorted_filenames:
            filepath = images_data[filename].get("filepath") # Verwende den gespeicherten Dateipfad
            prompt = images_data[filename].get("prompt", "Kein Prompt verfügbar")
            negative_prompt = images_data[filename].get("negative_prompt", "Kein negativer Prompt verfügbar")
            timestamp = images_data[filename].get("timestamp", "Unbekannt")

            if not os.path.exists(filepath):
                ctk.CTkLabel(self.gallery_scrollable_frame, text=f"Bilddatei nicht gefunden: {os.path.basename(filepath)} (Pfad: {filepath})", text_color="orange").pack()
                continue

            try:
                img = Image.open(filepath) # Korrigiert: Image.open
                img.thumbnail((200, 200), Image.LANCZOS)
                tk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(200, 200))

                img_frame = ctk.CTkFrame(self.gallery_scrollable_frame, corner_radius=8)
                img_frame.pack(pady=10, padx=10, fill="x", expand=True)
                img_frame.grid_columnconfigure(0, weight=1)

                img_label = ctk.CTkLabel(img_frame, image=tk_img, text="")
                img_label.image = tk_img
                img_label.grid(row=0, column=0, padx=10, pady=5)

                prompt_label = ctk.CTkLabel(img_frame, text=f"Prompt: {prompt}", wraplength=400, justify="left")
                prompt_label.grid(row=1, column=0, padx=10, pady=2, sticky="w")

                if negative_prompt:
                    negative_prompt_display_label = ctk.CTkLabel(img_frame, text=f"Negativ: {negative_prompt}", wraplength=400, justify="left", font=ctk.CTkFont(size=10), text_color="gray")
                    negative_prompt_display_label.grid(row=2, column=0, padx=10, pady=0, sticky="w")
                    timestamp_row = 3
                else:
                    timestamp_row = 2

                timestamp_label = ctk.CTkLabel(img_frame, text=f"Generiert: {timestamp}", font=ctk.CTkFont(size=10), text_color="gray")
                timestamp_label.grid(row=timestamp_row, column=0, padx=10, pady=2, sticky="w")

            except Exception as e:
                ctk.CTkLabel(self.gallery_scrollable_frame, text=f"Fehler beim Laden von {os.path.basename(filepath)}: {e}", text_color="red").pack()

    def _confirm_clear_all_images(self, gallery_window):
        """Fragt den Benutzer, ob alle Bilder gelöscht werden sollen."""
        response = messagebox.askyesno(
            "Alle Bilder löschen",
            "Möchten Sie wirklich ALLE generierten Bilder und deren Metadaten unwiderruflich löschen?\n\nDiese Aktion kann nicht rückgängig gemacht werden!"
        )
        if response:
            self._clear_all_images()
            gallery_window.destroy() # Schließt die Galerie nach dem Löschen
            self.update_status("Alle Bilder und Metadaten gelöscht.", "green")

    def _clear_all_images(self):
        """Löscht alle gespeicherten Bilder und die Metadatendatei."""
        if os.path.exists(IMAGE_DIR):
            for filename in os.listdir(IMAGE_DIR):
                file_path = os.path.join(IMAGE_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Fehler beim Löschen von Datei {file_path}: {e}")
        
        # Leere auch den Prompt-Verlauf
        self.prompt_history.clear()
        self._save_prompt_history()
        self._update_prompt_history_options()

        # Erstelle den Ordner neu, falls er gelöscht wurde (oder nur die Dateien darin)
        os.makedirs(IMAGE_DIR, exist_ok=True)
        # Erstelle die leere Metadatendatei neu
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)


if __name__ == "__main__":
    # Überprüfen, ob der /cpu-Parameter übergeben wurde
    force_cpu_mode = False
    if len(sys.argv) > 1 and "/cpu" in sys.argv:
        force_cpu_mode = True
        print("CPU-Modus erzwungen.")

    app = ImageGeneratorApp(force_cpu=force_cpu_mode)
    app.mainloop()
