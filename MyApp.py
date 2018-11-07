from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Rectangle
from kivy.graphics import Color
from kivy.core.window import Window

from kivy.properties import ObjectProperty

import datetime
import cv2

from Detector import Detector
from Reader import Reader

class KivyCapture(Image):
    def __init__(self, **kwargs):
        super(KivyCapture, self).__init__(**kwargs)
        self.capture = None
        self.fps = 30
        self.parent = None
        self.detector = Detector(graph='models/lp_detection_graph.pb', labels='license_plate_label_map.pbtxt')
        self.reader = Reader(graph='models/text_reading_graph.pb')
        self.running = ''

    def set_parent(self, parent):
        self.parent = parent

    def start(self):
        Clock.schedule_interval(self.update, 1.0 / self.fps)

    def pause(self):
        Clock.unschedule(self.update)

    # Clears the processed from the reader set and unschedule update
    def stop(self):
        Clock.unschedule(self.update)
        self.reader.processed_set = set()

    def update(self, dt):

        ret, frame = self.capture.read()
        if ret:
            # Object Detection
            frame, scores, num_detections, boxes = self.detector.detect(frame, resizing_factor=4)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            # Extracting license plates and read them
            if scores[0][0] > 0.995:
                height, width, channels = frame.shape
                # For All detected objects in the picture
                for i in range(int(num_detections[0])):
                    # Bounding box coordinates
                    ymin = int((boxes[0][i][0] * height))
                    xmin = int((boxes[0][i][1] * width))
                    ymax = int((boxes[0][i][2] * height))
                    xmax = int((boxes[0][i][3] * width))
                    lp_np = frame[ymin:ymax, xmin:xmax]
                    # Read text from license plate image
                    prediction, probability = self.reader.read(lp_np)
                    if probability > 0.95 and not self.reader.processed(prediction):
                        lp_buf1 = cv2.flip(lp_np, 0)
                        lp_buf = lp_buf1.tostring()
                        lp_image_texture = Texture.create(size=(lp_np.shape[1], lp_np.shape[0]), colorfmt='bgr')
                        lp_image_texture.blit_buffer(lp_buf, colorfmt='bgr', bufferfmt='ubyte')
                        # display image from the texture
                        record = Record()
                        record.update(image_texture=lp_image_texture, predicted_text=prediction,
                                      time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      coordinates='Latitude:\nLongitude')
                        self.parent.ids.data_grid.add_widget(record, len(self.parent.ids.data_grid.children))
                        self.parent.ids.scroll.scroll_to(record)

            self.texture = image_texture
            #self.parent.ids.video.texture = self.texture


class MainScreen(BoxLayout):
    pass


class Record(BoxLayout):
    def update(self, image_texture, predicted_text, time, coordinates):
        self.ids.lp_image.texture = image_texture
        self.ids.predicted_text.text = predicted_text
        self.ids.time.text = time
        self.ids.coordinates.text = coordinates

    def if_active(self):
        rec = None
        if self.ids.ch_box.active:
            with self.ids.lp_image.parent.canvas:
                Color(1, 1, 1, 0.5)
                self.rec  = Rectangle(pos=self.ids.lp_image.parent.pos, size=self.ids.lp_image.parent.size)
        if not self.ids.ch_box.active:
            self.ids.lp_image.parent.canvas.remove(self.rec)


class SButton(Button):
    pass


class MyApp(App):

    main_screen = None
    grid = None


    def build(self):
        self.title = 'License Plate Detection'
        self.main_screen = MainScreen()

        return self.main_screen

    def on_press_start(self):
        video = self.main_screen.ids.video
        if video.parent is not self.main_screen:
            video.set_parent(self.main_screen)
        if video.running is '': # First time
            video.capture = cv2.VideoCapture('c:/vids/GOPR0396.MP4')
            video.start()
            video.running = 'running'
        else:
            video.start()
        self.main_screen.ids.start_button.disabled = True
        self.main_screen.ids.pause_button.disabled = False

    def on_press_pause(self):
        video = self.main_screen.ids.video
        video.pause()
        video.running = 'paused'
        self.main_screen.ids.start_button.disabled = False
        self.main_screen.ids.pause_button.disabled = True

    def on_press_stop(self):
        self.main_screen.ids.video.stop()
        self.main_screen.ids.video.capture.release()
        self.main_screen.ids.video.running = ''
        self.main_screen.ids.video.texture = Texture.create(size=(1920, 1080), colorfmt='bgr')
        self.main_screen.ids.data_grid.clear_widgets()


        self.main_screen.ids.start_button.disabled = False
        self.main_screen.ids.pause_button.disabled = True

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.main_screen.capture.release()


if __name__ == '__main__':
    Window.size = (1920, 1080)
    MyApp().run()
