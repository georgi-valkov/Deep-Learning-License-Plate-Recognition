from kivy.config import Config, ConfigParser
Config.set('graphics', 'resizable', '1')
Config.set('graphics', 'top', '0')
Config.set('graphics', 'left', '0')
Config.set('graphics', 'position', 'custom')
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.core.window import Window
from kivy.properties import StringProperty, NumericProperty
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, RoundedRectangle, Color
from kivy.cache import Cache
from kivy.uix.settings import Settings
from threading import Thread, Event
import cv2
import numpy
import copy
import datetime
import os
import time
# Create a cache to store global variables and objects
# So they can be used across different screens and other classes
Cache.register('cache')


class BootScreen(Screen):

    boot_text = StringProperty('Initialization')
    init_text = StringProperty('Initializing...')
    title = 'Boot Screen'
    ready = False
    init_thread = None
    init_thread_stop_event = Event()

    def do_everything(self, dt=1):
        Clock.schedule_interval(self.boot, 1.0/3)
        self.init_thread = Thread(target=self.init_kivycapture, args=(1, self.init_thread_stop_event))
        self.init_thread.start()

    def init_kivycapture(self, *args):
        import sys
        import time
        from Detector import Detector
        from Reader import Reader
        main_screen = Cache.get('cache', 'MainScreen')
        # 0-Red, 1-Green, 2-Violet
        colors = ['[color=ff3333]', '[color=06c30c]', '[color=a15bf5]']
        self.update_init_text('\n', 'Checking all modules...')
        modules = list(set(sys.modules) & set(globals()))
        for module_name in modules:
            module = sys.modules[module_name]
            version = getattr(module, '__version__', 'unknown')
            if version is not 'unknown':
                self.update_init_text('\nImport - ', module_name + ' - ' + colors[2] + version + '[/color]')
            else:
                self.update_init_text('\nImport - ', module_name)

        self.update_init_text('\n', 'Import - TensorFlow - ' + colors[2] + '1.11.0[/color]')
        self.update_init_text('\n', 'Import - Detector')
        self.update_init_text('\n', 'Import - Reader')

        video = main_screen.ids.video
        self.update_init_text('\n', 'StartingDetector:')
        video.detector = Detector(graph='models/lp_detection_graph.pb', labels='license_plate_label_map.pbtxt')
        self.update_init_text(' ', colors[1] + 'Success [/color]')

        self.update_init_text('\n', 'Starting Reader:')
        video.reader = Reader(graph='models/text_reading_graph.pb')
        self.update_init_text(' ', colors[1] + 'Success [/color]')

        self.update_init_text('\n', 'Testing Reader:')
        video.reader.read(numpy.zeros((50,50,3),numpy.uint8))

        self.update_init_text(' ', colors[1] + 'Success [/color]')

        time.sleep(1)
        self.ready = True
        # Remove boot Text
        self.boot_text = ''
        # Remove Progress bar
        self.remove_widget(self.ids.progress_bar)

    def boot(self, *args):
        sm = Cache.get('cache', 'ScreenManager')
        if self.boot_text.count('.') == 5:
            self.boot_text = self.boot_text.replace('.....', '')
        if not self.ready:
            self.boot_text = self.boot_text + '.'
            # Update progress bar
            self.ids.progress_bar.value = self.ids.progress_bar.value + 60
        elif self.ready:
            self.ids.progress_bar.value = 1000
            # Stop init thread
            self.init_thread_stop_event.set()
            # Unschedule further boot updates
            Clock.unschedule(self.boot)
            # Switch to main screen
            sm.current = 'MainScreen'

    def update_init_text(self, delimiter, text):
        self.init_text = self.init_text + delimiter + text
    pass


class MainScreen(Screen):
    pass


class SettingsScreen(Screen):
    pass


class VideoFeedSettingsScreen(Screen):
    pass


class KivyCapture(Image):

    def __init__(self, **kwargs):
        super(KivyCapture, self).__init__(**kwargs)
        self.capture = None
        self.fps = int(App.get_running_app().config.get('video_settings', 'frames'))
        self.parent = None
        self.detector = None
        self.reader = None
        self.running = ''
        self.texture = Texture.create(size=(1920, 1080), colorfmt='bgr')
        self.detection_certainty = int(App.get_running_app().config.get('detector', 'detection_certainty'))
        self.reading_certainty = int(App.get_running_app().config.get('reader', 'reading_certainty'))
        self.list = []

    # Counts non overlapping chars in two strings
    def non_overlap(self, string1, string2):
        count = 0
        for i in range(min(len(string1), len(string2))):
            if string1[i] != string2[i]:
                count = count + 1
        return count

    # Calls update function with a clock
    def start(self):
        Clock.schedule_interval(self.update, 1.0 / self.fps)

    # Unschedule the update function
    def pause(self):
        Clock.unschedule(self.update)

    # Clears the processed from the reader set and unschedule update
    def stop(self):
        Clock.unschedule(self.update)
        self.reader.processed_set = set()

    # Before calling this method capture must be set to cv2.VideoCapture object by a button
    def update(self, dt):
        main_screen = Cache.get('cache', 'MainScreen')
        ret, frame = self.capture.read()
        # if frame gotten successfully
        if ret:
            # Create a copy of original frame
            clear_frame = copy.deepcopy(frame)
            # Object Detection
            frame, scores, num_detections, boxes = self.detector.detect(frame, resizing_factor=4)
            # convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

            for i in range(int(num_detections[0])):
                # Extracting license plates and read them
                if scores[0][i] > (self.detection_certainty / 100):
                    height, width, channels = frame.shape
                    # For All detected objects in the picture
                    # Bounding box coordinates
                    ymin = int((boxes[0][i][0] * height))
                    xmin = int((boxes[0][i][1] * width))
                    ymax = int((boxes[0][i][2] * height))
                    xmax = int((boxes[0][i][3] * width))
                    lp_np = clear_frame[ymin:ymax, xmin:xmax]
                    # Read text from license plate image
                    prediction, probability = self.reader.read(lp_np)
                    if probability > (self.reading_certainty / 100):

                        if not self.list:
                            self.list.append([prediction, probability, lp_np])
                        else:
                            element = self.list[len(self.list) - 1]
                            # Probably same picture but different prediction
                            if self.non_overlap(prediction, element[0]) <= 2:
                                self.list.append([prediction, probability, lp_np])
                            # Different picture get out prediction with highest probability
                            # TODO: Get out best prediction at the end of the video since there won't be any other to push it out (use frame count)
                            else:
                                probability = 0
                                for p in self.list:
                                    if p[1] > probability:
                                        prediction = p[0]
                                        probability = p[1]
                                        lp_np = p[2]

                                if not self.reader.processed(prediction):
                                    lp_buf1 = cv2.flip(lp_np, 0)
                                    lp_buf = lp_buf1.tostring()
                                    lp_image_texture = Texture.create(size=(lp_np.shape[1], lp_np.shape[0]), colorfmt='bgr')
                                    lp_image_texture.blit_buffer(lp_buf, colorfmt='bgr', bufferfmt='ubyte')
                                    # TODO: Query database to fill out make_model, parking_permit, valid
                                    # display image from the texture
                                    record = Record()
                                    record.update(image_texture=lp_image_texture, predicted_text=prediction,
                                                  time=datetime.datetime.now().strftime("%Y-%m-%d\n%H:%M:%S"),
                                                  make_model='Ford\nTaurus',
                                                  parking_permit='West\nW00013332',
                                                  valid='Yes')
                                    main_screen.ids.data_grid.add_widget(record, len(main_screen.ids.data_grid.children))
                                    main_screen.ids.scroll.scroll_to(record)

                                    self.reader.processed_set.add(prediction)
                                self.list = []

            self.texture = image_texture
        # TODO: Decide if stop needs to be invoked after video ends
        # else:
        #     App.get_running_app().on_press_stop()


class Record(BoxLayout):
    def update(self, image_texture, predicted_text, time, make_model, parking_permit, valid):
        self.ids.lp_image.texture = image_texture
        self.ids.predicted_text.text = predicted_text
        self.ids.time.text = time
        self.ids.make_model.text = make_model
        self.ids.parking_permit.text = parking_permit
        self.ids.valid.text = valid

    def if_active(self):
        rec = None
        if self.ids.ch_box.active:
            with self.ids.lp_image.parent.canvas:
                Color(1, 1, 1, 0.5)
                self.rec = RoundedRectangle(pos=(self.ids.lp_image.parent.pos[0], self.ids.lp_image.parent.pos[1] - 5),
                                            size=(self.ids.lp_image.parent.size[0], self.ids.lp_image.parent.size[1] + 10),
                                            radius=[20,])
        if not self.ids.ch_box.active:
            self.ids.lp_image.parent.canvas.remove(self.rec)
    # TODO: Make a function that wraps a text in a color

class Settings(Settings):

    # Close settings
    def on_close(self):
        # Get screen manager
        sm = Cache.get('cache', 'ScreenManager')
        # Switch to main screen
        sm.current = 'MainScreen'

    def on_config_change(self, config, section,
                         key, value):
        main_screen = Cache.get('cache', 'MainScreen')

        if key == 'frames':
            main_screen.ids.video.fps = int(value)
        elif key == 'detection_certainty':
            main_screen.ids.video.detection_certainty = int(value)
        elif key == 'reading_certainty':
            main_screen.ids.video.reading_certainty = int(value)
        #print (config, section, key, value)

class MyApp(App):
    main_screen = None
    settings_screen = None
    def build(self):
        self.settings_cls = Settings
        self.use_kivy_settings = False

        # Create the screen manager
        tr = FadeTransition(duration=1.)
        self.sm = ScreenManager(transition=tr)

        screens = [BootScreen(name='BootScreen'), MainScreen(name='MainScreen'), SettingsScreen(name='SettingsScreen'),
                   VideoFeedSettingsScreen(name='VideoFeedSettingsScreen')]
        for screen in screens:
            self.sm.add_widget(screen)
        # Append objects to the cache so they can be used across the classes
        Cache.append('cache', 'ScreenManager', self.sm)
        Cache.append('cache', 'BootScreen', screens[0])
        Cache.append('cache', 'MainScreen', screens[1])
        Cache.append('cache', 'SettingsScreen', screens[2])

        self.main_screen = screens[1]
        self.settings_screen = screens[2]
        self.title = 'License Plate Recognition'

        # Window size comes from config file and it's the last set width height in settings
        Window.size = (int(self.config.get('appearance', 'window_width')),
                       int(self.config.get('appearance', 'window_height')))

        return self.sm

    # Build config
    def build_config(self, config):
        config.setdefaults('appearance', {
            'window_height': 1080,
            'window_width': 1920,
        })
        config.setdefaults('video_settings', {
            'frames': 30
        })
        config.setdefaults('detector', {
            'detection_certainty': 90,
        })
        config.setdefaults('reader', {
            'reading_certainty': 95,
        })
        # Put config object into the cache so it can be used outside
        Cache.append('cache', 'config', config)

    # Build settings from file
    def build_settings(self, settings):
        settings.add_json_panel('Settings',
                                self.config,
                                'settings.json')

    def on_press_start(self):
        video = self.main_screen.ids.video
        if video.running is '': # First time
            video.capture = cv2.VideoCapture('20181119_144837.mp4')
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
        if self.main_screen.ids.video.capture is not None:
            self.main_screen.ids.video.stop()
            self.main_screen.ids.video.capture.release()
            self.main_screen.ids.video.running = ''
            self.main_screen.ids.video.texture = Texture.create(size=(1920, 1080), colorfmt='bgr')
            self.main_screen.ids.data_grid.clear_widgets()

            self.main_screen.ids.start_button.disabled = False
            self.main_screen.ids.pause_button.disabled = True

    def on_stop(self):
        # Save current window size on close
        width, height = Window.size
        config = Cache.get('cache', 'config')
        config.set('appearance', 'window_width', width)
        config.set('appearance', 'window_height', height)
        config.write()

        # without this, app will not exit even if the window is closed
        if self.main_screen.ids.video.capture is not None:
            self.main_screen.ids.video.capture.release()


if __name__=='__main__':
    MyApp().run()
