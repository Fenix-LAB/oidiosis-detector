import torch
import numpy as np
import cv2
import time


class ObjectDetection:
    """
    Deteccion de objetos en tiempo real con YOLOv7, deteccion de odiosis en arandanos.
    """
    
    def __init__(self):
        """
        Contructor de la clase.
        Inicia el modelo y configura el hardware.
        """
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)



    def load_model(self):
        """
        Caraga el modelo de deteccion de objetos.
        :return: Modelo de deteccion de objetos.
        """
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'blueberry_model.pt',
                        force_reload=False, trust_repo=True)
        
        model.conf = 0.1 # confidence threshold (0-1)
        return model


    def score_frame(self, frame):
        """
        Toma un frame como entrada y devuelve las etiquetas y las coordenadas de los objetos detectados por el modelo en el frame.
        :param frame: Frame que se va a puntuar en formato numpy.
        :return: Etiquetas y coordenadas de los objetos detectados por el modelo en el frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Tomando las etiquetas y las coordenadas de los objetos detectados por el modelo en el frame, traza las cajas delimitadoras y las etiquetas en el frame.
        :param results: Etiquetas y coordenadas de los objetos detectados por el modelo en el frame.
        :param frame: Frame en formato numpy.
        :return: Frame con las cajas delimitadoras y las etiquetas trazadas.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame


    def __call__(self):
        """
        Ejecuta la deteccion de objetos en tiempo real.
        El metodo __call__ permite que el objeto sea llamado como una funcion.
        :return: void
        """
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            cv2.imshow("img", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        # close all windows
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create a new object and execute.
    detection = ObjectDetection()
    detection()