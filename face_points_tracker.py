import base64
import csv
import os
import re
from pathlib import Path
from uuid import uuid4

import cv2
import dash
import dash_bootstrap_components as dbc
import mediapipe as mp
import numpy as np
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Path to Desktop
desktop_path = Path.home() / "Bureau"

# --------------------- Helper Functions ---------------------
def calculate_angle(pointA, pointB, pointC):
    # Calcul des longueurs des côtés du triangle
    a = calculate_distance(pointB, pointC)
    b = calculate_distance(pointA, pointC)
    c = calculate_distance(pointA, pointB)

    # Utilisation de la loi des cosinus pour calculer l'angle au point A
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

    # Convertir l'angle en degrés
    angle_degrees = np.degrees(angle)
    return angle_degrees

def calculate_area_of_triangle(point1, point2, point3):
    # Calculate the area of a triangle given three points
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def calculate_area_of_quadrilateral(point1, point2, point3, point4):
    # Diviser le quadrilatère en deux triangles et calculer l'aire de chaque
    area1 = calculate_area_of_triangle(point1, point2, point3)
    area2 = calculate_area_of_triangle(point1, point3, point4)
    return area1 + area2

def calculate_distance(point1, point2):
    # Calculate the distance between two points
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def draw_cross(img, center, color=(0, 255, 0), size=10):
    # Draw a cross on an image at the specified center point
    cv2.line(img, (center[0] - size, center[1]), (center[0] + size, center[1]), color, 2)
    cv2.line(img, (center[0], center[1] - size), (center[0], center[1] + size), color, 2)

def decode_and_save_video(video_contents, filename):
    # Decode base64 video and save it
    content_type, content_string = video_contents.split(',')
    decoded = base64.b64decode(content_string)
    video_path = desktop_path / f"temp_{uuid4()}.mp4"
    with open(video_path, 'wb') as f:
        f.write(decoded)
    return video_path

def process_video(video_path, labels, landmarks, csv_name):
    # Process video to track face landmarks and write results to CSV
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "Error opening video."

    csv_path = desktop_path / f"{csv_name}.csv"
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        headers = generate_csv_headers(labels, landmarks)
        csv_writer.writerow(headers)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            write_frame_data_to_csv(image, face_mesh, csv_writer, labels, landmarks)

        cap.release()
    return csv_path

def generate_csv_headers(labels, landmarks):
    # Generate CSV headers based on label and landmark information
    headers = []
    for label, landmark_group in zip(labels, landmarks):
        if len(landmark_group) == 4:
            headers.append(f'{label}_area')
        elif len(landmark_group) == 3:
            headers.append(f'{label}_angle')
        elif len(landmark_group) == 2:
            headers.append(f'{label}_distance')
        else:
            headers.extend([f'{label}_x', f'{label}_y'])
    return headers

def write_frame_data_to_csv(image, face_mesh, csv_writer, labels, landmarks):
    # Write data for a single frame to the CSV
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            row = process_landmarks_for_row(face_landmarks, labels, landmarks, image)
            csv_writer.writerow(row)

def process_landmarks_for_row(face_landmarks, labels, landmarks, image):
    # Process face landmarks to generate a row of data
    row = []
    for label, landmark_group in zip(labels, landmarks):
        if len(landmark_group) == 4:
            points = [(face_landmarks.landmark[id].x * image.shape[1],
                       face_landmarks.landmark[id].y * image.shape[0])
                      for id in landmark_group]
            area = calculate_area_of_quadrilateral(*points)
            row.append(area)
        elif len(landmark_group) == 3:
            points = [(face_landmarks.landmark[id].x * image.shape[1],
                       face_landmarks.landmark[id].y * image.shape[0])
                      for id in landmark_group]
            angle = calculate_angle(points[0], points[1], points[2])
            row.append(angle)
        elif len(landmark_group) == 2:
            points = [(face_landmarks.landmark[id].x * image.shape[1],
                       face_landmarks.landmark[id].y * image.shape[0])
                      for id in landmark_group]
            distance = calculate_distance(*points)
            row.append(distance)
        else:
            for id in landmark_group:
                landmark = face_landmarks.landmark[id]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                row.extend([x, y])
    return row

def remove_tracking_point(children, ids, index_to_remove):
    # Remove a tracking point based on its index
    return [child for child, id in zip(children, ids) if str(id['index']) != str(index_to_remove)]

def create_tracking_point_row(index):
    # Create a new tracking point row for the UI
    return dbc.Row([
        dbc.Col(dbc.Checklist(options=[{"label": " Visage stabilisé", "value": 1}],
                              value=[], id={'type': 'cancel-head', 'index': index}, switch=True), width=3),
        dbc.Col(dcc.Input(id={'type': 'dynamic-label', 'index': index}, type='text', placeholder='Titre de colonne'), width=3),
        dbc.Col(dcc.Input(id={'type': 'dynamic-landmark', 'index': index}, type='text',
                          placeholder='Point(s)      (ex: 21,32)'), width=3),
        dbc.Col(dbc.Button('X', id={'type': 'remove-point', 'index': index}, n_clicks=0, className="btn btn-danger"),
                width=3)
    ], className='d-flex align-items-center', style={'margin-top': '10px'})

# --------------------- Layout ---------------------
# Function to create initial tracking points
def initial_tracking_points():
    children = [create_tracking_point_row(0)]  # Create the first tracking point
    return children

app.layout = html.Div([
    dbc.Container([
        dbc.Row(dbc.Col(html.H1("Tracking Modalities"))),
        dbc.Row(dbc.Col(dcc.Upload(id='upload-video',
                                   children=html.Div(['Glisser-déposer ou ', html.A('choisir une vidéo')]),
                                   style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                                          'margin': '10px'}, multiple=False))),
        dbc.Row(dbc.Col(html.Div(id='video-name'))),
        dbc.Row(dbc.Col(dcc.Input(id='csv-name-input', type='text', placeholder='Fichier CSV à créer',
                                  style={'margin-top': '10px'}))),
        html.Div(id='tracking-points-container', children=initial_tracking_points()),
        dbc.Button("Ajouter un point", id="add-point", n_clicks=0, className="me-1",
                   style={'margin-top': '10px'}),
        html.Hr(),
        dbc.Row(dbc.Col(dbc.Button('Démarrer', id='run-tracking', n_clicks=0, color="danger",
                                   style={'margin-top': '20px', 'font-size': '20px', 'padding': '2px 24px'}))),
        dbc.Row(dbc.Col(html.Div(id='output-state'))),


        html.Hr(),  # Ligne de séparation
        html.H2("Comment utiliser ce script"),
        html.P("Pour utiliser ce script de suivi, veuillez suivre les étapes suivantes :"),
        html.Ol([
            html.Li("Téléchargez une vidéo en utilisant le bouton 'Sélectionner une Vidéo'."),
            html.Li("Ajoutez un point de suivi en cliquant sur 'Ajouter un Point de Suivi'."),
            html.Li("Pour chaque point de suivi :"),
            html.Ul([
                html.Li("Utilisez un seul point pour suivre une position spécifique sur le visage."),
                html.Li("Utilisez deux points pour mesurer la distance entre deux positions sur le visage."),
                html.Li("Utilisez trois points pour calculer l'angle formé par ces trois positions sur le visage."),
                html.Li("Utilisez quatres points pour calculer l'aire délimitée par ces quatres positions sur le visage."),
            ]),
            html.Li("Entrez un nom pour le fichier CSV où les données seront enregistrées."),
            html.Li("Cliquez sur 'Démarrer le Suivi' pour traiter la vidéo et enregistrer les résultats.")
        ]),
        html.P("Assurez-vous que tous les champs sont correctement remplis avant de démarrer le suivi."),
    ])
])


# --------------------- Callbacks ---------------------

@app.callback(
    Output('tracking-points-container', 'children'),
    [Input('add-point', 'n_clicks'),
     Input({'type': 'remove-point', 'index': ALL}, 'n_clicks')],
    [State('tracking-points-container', 'children'),
     State({'type': 'remove-point', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_tracking_points(add_clicks, remove_clicks_list, children, ids):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks yet'

    if 'add-point' in button_id:
        new_index = len(children)
        children.append(create_tracking_point_row(new_index))
    elif 'remove-point' in button_id:
        index_to_remove = int(re.search(r'\d+', ctx.triggered[0]['prop_id']).group())
        children = remove_tracking_point(children, ids, index_to_remove)

    return children

@app.callback(
    Output('video-name', 'children'),
    Input('upload-video', 'filename')
)
def update_video_name(filename):
    return f"Video uploaded: {filename}" if filename else "Aucune vidéo sélectionnée"

@app.callback(
    Output('output-state', 'children'),
    [Input('run-tracking', 'n_clicks')],
    [State('upload-video', 'contents'), State('upload-video', 'filename'),
     State({'type': 'dynamic-label', 'index': ALL}, 'value'),
     State({'type': 'dynamic-landmark', 'index': ALL}, 'value'),
     State('csv-name-input', 'value')]
)
def update_output(n_clicks, video_contents, filename, labels, landmarks, csv_name):
    if n_clicks <= 0 or video_contents is None or filename is None or not labels or not landmarks or not csv_name:
        return "Merci de compléter votre demande."

    video_path = decode_and_save_video(video_contents, filename)
    landmarks = [[int(id.strip()) for id in landmark_group.split(',') if id.strip().isdigit()] for landmark_group in landmarks]

    csv_path = process_video(video_path, labels, landmarks, csv_name)
    os.remove(video_path)  # Clean up by removing the temporary video file

    return f"Le suivi est terminé et enregistré dans le fichier : {csv_path}."

if __name__ == '__main__':
    app.run_server(debug=True)
