#!/usr/bin/env python3
import os
import cv2
import sys
import dlib
import argparse
import glob
import numpy as np

import tkinter as tk
from tkinter import filedialog
import pygame
import pygame_gui
#import Face Recognition libraries
import mediapipe as mp

# helper modules
from drawFace import draw
import reference_world as world

#Settingup MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--focal",
                    type=float,
                    help="Callibrated Focal Length of the camera")
# parser.add_argument("-v", "--videosource", type=str, default=None,
# 	help="Enter the video file path")

args = vars(parser.parse_args())

face3Dmodel = world.ref3DModel()

def upload_action():
    # first create a pygame window that will ask the user to upload a video and then pass the video to the main function
    pygame.init()
    screen = pygame.display.set_mode((720, 720))
    pygame.display.set_caption("Upload a video")
    screen.fill((217, 219, 241))
    # create a font object.
    font = pygame.font.Font('Poppins-Regular.ttf', 32)
    sfont = pygame.font.Font('Poppins-Regular.ttf', 16)
    font2 = pygame.font.Font(None, 32)
    # create a text surface object,
    # on which text is drawn on it.
    text = sfont.render('Upload a video', True, (0, 0, 0))
    # create a rectangular object for the
    # text surface object
    textRect = text.get_rect()
    # set the center of the rectangular object.
    textRect.center = (360, 360)
    screen.blit(text, textRect)
    pygame.display.update()

    text = font.render('Spectrum Sense', True, (0, 0, 0))
    # create a rectangular object for the
    # text surface object
    textRect = text.get_rect()
    # set the center of the rectangular object.
    textRect.center = (360, 320)
    screen.blit(text, textRect)
    pygame.display.update()
    running = True
    # get the logo (2).png file from /images folder
    logo = pygame.image.load('images/logo (2).png')
    # set the logo position
    # resize the logo to be a bit smaller
    logo = pygame.transform.scale(logo, (200, 200))
    logoRect = logo.get_rect()
    logoRect.center = (360, 180)
    screen.blit(logo, logoRect)
    pygame.display.update()
    # create a button that when clicked will open a file dialog to upload a video
    buttonRect = pygame.Rect(300, 400, 120, 50)
    buttonRectHover = pygame.Rect(300, 400, 120, 50)
    buttonText = font2.render('Upload', True, (255, 255, 255))
    screen.blit(buttonText, (320, 415))
    pygame.display.update()
    while running:
        mouse_pos = pygame.mouse.get_pos()
        if buttonRect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, (87, 93, 130), buttonRect)
            screen.blit(buttonText, (320, 415))
        else:
            pygame.draw.rect(screen, (125, 132, 178), buttonRect)
            screen.blit(buttonText, (320, 415))
        pygame.display.update()

        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if buttonRect.collidepoint(mouse_pos):
                    root = tk.Tk()
                    root.withdraw()
                    video_file = filedialog.askopenfilename()
                    main(video_file)
                    running = False
    pygame.quit()



def main(video_file):
    pygame.init()
    screen = pygame.display.set_mode((840, 720))
    screen.fill((217, 219, 241))
    pygame.display.set_caption("Spectrum Sense")
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    cap = cv2.VideoCapture(video_file)
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frontFrames = 0

    # put the logo in the top right corner of the pygame window
    logo = pygame.image.load('images/logo (2).png')
    logo = pygame.transform.scale(logo, (100, 100))
    logoRect = logo.get_rect()
    logoRect.topleft = (720, 40)
    screen.blit(logo, logoRect)

    # text under the logo
    font = pygame.font.Font('Poppins-Regular.ttf', 16)
    text = font.render('Spectrum Sense', True, (0, 0, 0))
    textRect = text.get_rect()
    textRect.topleft = (700, 140)
    screen.blit(text, textRect)

    # use a light font for the text
    font = pygame.font.Font('Poppins-Regular.ttf', 26)

    # font for button 
    font2 = pygame.font.Font(None, 32)

    pygame.draw.circle(screen, (0, 0, 0), (420, 430), 20)
    autismlabel = font.render('Risk of Autism', True, (0, 0, 0))
    textRect = autismlabel.get_rect()
    textRect.center = (580, 430)
    screen.blit(autismlabel, textRect)

    while (cap.isOpened()):
        GAZE = "Face Not Found"
        ret, img = cap.read()
        pyimg = img
        # if the video is still running, resize the image and display it on the pygame window else display the result
        if ret:
            # get the size of the image divided by 2 and resize the image
            pyimg = cv2.resize(pyimg, (int(pyimg.shape[1] / 5), int(pyimg.shape[0] / 5)))
            # rotate the image by 90 degrees
            pyimg = cv2.rotate(pyimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
            pyimg = cv2.cvtColor(pyimg, cv2.COLOR_BGR2RGB)
            pyimg = pygame.surfarray.make_surface(pyimg)
            screen.blit(pyimg, (30, 30))
            pygame.display.update()
        else:
            # show these statistics on the pygame window frames processed, frames in front, and frames at the back, and the result under the video
            text1 = font.render(f'Frames Processed: {nFrames}', True, (0, 0, 0))
            textRect1 = text1.get_rect()
            textRect1.topleft = (30, 430)
            screen.blit(text1, textRect1)
            text2 = font.render(f'Frames in front: {frontFrames}', True, (0, 0, 0))
            textRect2 = text2.get_rect()
            textRect2.topleft = (30, 470)
            screen.blit(text2, textRect2)
            text3 = font.render(f'Frames at back: {nFrames - frontFrames}', True, (0, 0, 0))
            textRect3 = text3.get_rect()
            textRect3.topleft = (30, 510)
            screen.blit(text3, textRect3)


        if not ret:
            if frontFrames < int(nFrames * 0.5):
                # change the color of the circle to red if the patient is at risk of having Autism Spectrum Disorder
                pygame.draw.circle(screen, (255, 0, 0), (420, 430), 20)
                additionalNotes1 = font.render(f'This person is at a risk of autism', True, (0, 0, 0))
                additionalNotes2 = font.render(f'consider contacting a doctor', True, (0, 0, 0))
                additionalNotes3 = font.render(f'for further evaluation.', True, (0, 0, 0))
                textRect1 = additionalNotes1.get_rect()
                textRect1.topleft = (400, 500)
                screen.blit(additionalNotes1, textRect1)
                textRect2 = additionalNotes2.get_rect()
                textRect2.topleft = (400, 540)
                screen.blit(additionalNotes2, textRect2)
                textRect3 = additionalNotes3.get_rect()
                textRect3.topleft = (400, 580)
                screen.blit(additionalNotes3, textRect3)

                # create analyze another button
                analyzeAnotherButton = pygame.Rect(400, 660, 200, 50)
                analyzeAnotherText = font2.render('Analyze Another', True, (255, 255, 255))
                analyzeAnotherTextRect = analyzeAnotherText.get_rect()
                analyzeAnotherTextRect.center = analyzeAnotherButton.center
                screen.blit(analyzeAnotherText, analyzeAnotherTextRect)

                pygame.display.update()

                running = True
                while running:
                    # check if the analyze another button is clicked
                    for event in pygame.event.get():
                        # hover use the colors (87, 93, 130) and (125, 132, 178)
                        if analyzeAnotherButton.collidepoint(pygame.mouse.get_pos()):
                            pygame.draw.rect(screen, (87, 93, 130), analyzeAnotherButton)
                            screen.blit(analyzeAnotherText, analyzeAnotherTextRect)
                        else:
                            pygame.draw.rect(screen, (125, 132, 178), analyzeAnotherButton)
                            screen.blit(analyzeAnotherText, analyzeAnotherTextRect)
                        pygame.display.update()

                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if analyzeAnotherButton.collidepoint(event.pos):
                                upload_action()
                                running = False
                                break
                        elif event.type == pygame.QUIT:
                            running = False
                            break

            else:
                # change the color of the circle to green if the patient is not at risk of having Autism Spectrum Disorder
                pygame.draw.circle(screen, (0, 255, 0), (420, 430), 20)
                additionalNotes1 = font.render(f'This person is not at a risk of', True, (0, 0, 0))
                additionalNotes2 = font.render(f' autism. No further evaluation ', True, (0, 0, 0))
                additionalNotes3 = font.render(f'is required.', True, (0, 0, 0))
                textRect1 = additionalNotes1.get_rect()
                textRect1.topleft = (400, 500)
                screen.blit(additionalNotes1, textRect1)
                textRect2 = additionalNotes2.get_rect()
                textRect2.topleft = (400, 540)
                screen.blit(additionalNotes2, textRect2)
                textRect3 = additionalNotes3.get_rect()
                textRect3.topleft = (400, 580)
                screen.blit(additionalNotes3, textRect3)

                # create analyze another button
                analyzeAnotherButton = pygame.Rect(400, 660, 200, 50)
                pygame.draw.rect(screen, (0, 0, 255), analyzeAnotherButton)
                analyzeAnotherText = font2.render('Analyze Another', True, (255, 255, 255))
                analyzeAnotherTextRect = analyzeAnotherText.get_rect()
                analyzeAnotherTextRect.center = analyzeAnotherButton.center
                screen.blit(analyzeAnotherText, analyzeAnotherTextRect)

                pygame.display.update()
                running = True
                while running:
                    # check if the analyze another button is clicked
                    for event in pygame.event.get():
                        # hover use the colors (87, 93, 130) and (125, 132, 178)
                        if analyzeAnotherButton.collidepoint(pygame.mouse.get_pos()):
                            pygame.draw.rect(screen, (87, 93, 130), analyzeAnotherButton)
                            screen.blit(analyzeAnotherText, analyzeAnotherTextRect)
                        else:
                            pygame.draw.rect(screen, (125, 132, 178), analyzeAnotherButton)
                            screen.blit(analyzeAnotherText, analyzeAnotherTextRect)
                        pygame.display.update()

                        print("Event noticed")
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            print("Button clicked")
                            if analyzeAnotherButton.collidepoint(event.pos):
                                upload_action()
                                running = False  # Stop the loop when the button is clicked
                                break
                        elif event.type == pygame.QUIT:
                            running = False  # Stop the loop when the Pygame window is closed
                            break

                # when the button is clicked, close the pygame window and open the upload_action function
                pygame.display.update()
                running = True
                while running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                pygame.quit()
                break
        image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not ret:
            print(f'[ERROR - System]Cannot read from source: {args["camsource"]}')
            break

        if results.detections:
            for detection in results.detections:
                location = detection.location_data
                relative_bounding_box = location.relative_bounding_box
                x_min = relative_bounding_box.xmin
                y_min = relative_bounding_box.ymin
                widthh = relative_bounding_box.width
                heightt = relative_bounding_box.height


                absx,absy=mp_drawing._normalized_to_pixel_coordinates(x_min,y_min,w,h)
                abswidth,absheight = mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h)
                
            newrect = dlib.rectangle(absx,absy,abswidth,absheight)
            cv2.rectangle(image, (absx, absy), (abswidth, absheight),
            (0, 255, 0), 2)
            shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

            draw(image, shape)

            refImgPts = world.ref2dImagePoints(shape)

            height, width, channels = img.shape
            focalLength = 50 * width
            cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

            mdists = np.zeros((4, 1), dtype=np.float64)

            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(
                face3Dmodel, refImgPts, cameraMatrix, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

            #  draw nose line
            p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(image, p1, p2, (110, 220, 0),
                     thickness=2, lineType=cv2.LINE_AA)

            # calculating euler angles
            rmat, jac = cv2.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            print('*' * 80)
            # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
            z = np.arctan2(Qz[0][0], Qz[1][0])
            # print("ThetaX: ", x)
            print("ThetaY: ", y)
            # print("ThetaZ: ", z)
            print('*' * 80)
            if angles[1] < -15:
                GAZE = "Looking: Left"
            elif angles[1] > 15:
                GAZE = "Looking: Right"
            else:
                GAZE = "Looking: Forward"
                frontFrames = frontFrames + 1


    cap.release()
    cv2.destroyAllWindows()

    # when the x button is clicked, close the pygame window
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    upload_action()