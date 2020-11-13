import os
import cv2
import pickle
import numpy as np
import face_recognition as fr


class FaceRecognizer:

    def __init__(self):
        self.names = []
        self.encodings = []

    def recognize(self, image, frame_thickness=3):
        results = []

        locations = fr.face_locations(image)
        encodings = fr.face_encodings(image)

        for encoding, location in zip(encodings, locations):
            try:
                distances = fr.face_distance(self.encodings, encoding)
                match_idx = np.argmin(distances)

                if distances[match_idx] < 0.4:
                    match = self.names[match_idx]
                    results.append(match)
                else:
                    match = "no match"

                print(self.names)
                print(distances)
                print(f"Match: {match}")

                left_top = (location[3], location[0])
                right_bottom = (location[1], location[2])

                color = [0, 255, 0]
                image = cv2.rectangle(img=image, pt1=left_top, pt2=right_bottom,
                                      color=color, thickness=frame_thickness)
            except:
                pass

        return results, image

    def load_encodings(self, base_dir):
        for name in os.listdir(base_dir):
            face_dir = f"{base_dir}/{name}/"
            idx = 0
            while True:
                img_path = face_dir + f"image_{idx}.jpg"
                pkl_path = face_dir + f"encoding_{idx}.pkl"
                if not os.path.isfile(img_path):
                    break

                image = fr.load_image_file(img_path)
                # load encoding if exists
                if os.path.isfile(pkl_path):
                    print(f'Loading encoding of {name}/image_{idx}.jpg...')

                    f = open(pkl_path, 'rb')
                    face_encoding = pickle.load(f)
                    f.close()

                    print('Done!')
                # create encoding file if not exists
                else:
                    print(f'Creating encoding of {name}/image_{idx}.jpg...')

                    face_encoding = fr.face_encodings(image)[0]
                    f = open(pkl_path, "wb")
                    pickle.dump(face_encoding, f)
                    f.close()

                    print('Done!')

                self.encodings.append(face_encoding)
                self.names.append(name)

                idx += 1
