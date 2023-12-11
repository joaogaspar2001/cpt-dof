"""
Filename: ui.py
Author: João Marafuz Gaspar
Date Created: 21-Sep-2023
Description: This module defines the UserInterface class for the Computational Depth-of-Field application,
             which manages the GUI, handling user inputs, displaying images, and interacting with
             other functionalities to apply depth-of-field effects to loaded images.
"""


import os
import cv2

from PyQt5.QtCore import Qt, QPoint, QRect, QUrl
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QDesktopWidget,
    QHBoxLayout,
    QMainWindow,
    QSlider,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from depth_map import DepthMapGenerator
from depth_map_CNN import DepthMapCNNGenerator
from bilateral_filter import BilateralFilter
from ken_burns import KenBurnsEffect


class UserInterface(QMainWindow):
    def __init__(self, args):
        """
        Initialize the UserInterface instance, setting up attributes and UI elements.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        super().__init__()
        self.args = args
        self.initAttributes()
        self.initUI()

    def initAttributes(self):
        """
        Initialize instance attributes, including file paths, state, pixmap objects,
        drawing settings, and aperture size.
        """
        self.inputFilePath = None
        self.currentFilePath = None
        self.currentState = None
        self.currentPixmap = None
        self.scribblePixmap = None

        self.drawing = False
        self.lastPoint = None

        self.penThickness = 1
        self.penColor = QColor(0, 0, 0)

        self.focus_depth = None
        self.apertureSize = 2.8

        self.frames = []

        self.videoPlayer = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.videoOutput = QVideoWidget(self)
        self.videoOutput.setAspectRatioMode(Qt.KeepAspectRatio)

    def initUI(self):
        """
        Initialize the user interface, setting up windows, labels, layouts, buttons,
        and sliders. Optionally loads a default image if "-d" flag is passed as a
        command-line argument.
        """
        self._setupWindow()
        self._setupInstructionLabel()
        self._setupImageLabel()
        self._setupButtons()
        self._setupSliders()
        self._organizeLayouts()
        self.show()

        # If an image file argument was provided, load it
        if self.args.file:
            self.loadImage(default_image=True)

    def _setupWindow(self):
        """
        Configure main window properties, including geometry, title, and initial position.
        """
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("Computational Depth-of-Field")

        screen_res = QDesktopWidget().screenGeometry()
        x = int((screen_res.width() - self.frameSize().width()) / 2)
        y = int((screen_res.height() - self.frameSize().height()) / 2)
        self.move(x, y)

    def _setupInstructionLabel(self):
        """
        Create and configure a QLabel to display instructions above the image label.
        """
        self.instructionLabel = QLabel(self)
        self.instructionLabel.setAlignment(Qt.AlignCenter)
        self.instructionLabel.setFixedHeight(50)
        self.instructionLabel.setWordWrap(True)
        self.instructionLabel.setText(
            "<b>Welcome!</b> Please <b>load</b> an image to start."
        )

    def _setupImageLabel(self):
        """
        Create and configure a QLabel to display images, ensuring a minimum size and
        center alignment.
        """
        self.imageLabel = QLabel(self)
        self.imageLabel.setMinimumSize(200, 200)  # Prevents window too small
        self.imageLabel.setAlignment(Qt.AlignCenter)

        # Set video as parent of the image label
        self.videoOutput.setParent(self.imageLabel)

    def _setupButtons(self):
        """
        Initialize UI buttons and connect them to their respective slots (functions).
        """
        self.loadImageButton = QPushButton("Load Image", self)
        self.loadImageButton.clicked.connect(self.loadImage)

        self.depthMapButton = QPushButton("Depth Map", self)
        self.depthMapButton.clicked.connect(self.depthMap)

        self.depthMapCNNButton = QPushButton("Depth Map CNN", self)
        self.depthMapCNNButton.clicked.connect(self.depthMapCNN)

        self.filterButton = QPushButton("Filter", self)
        self.filterButton.clicked.connect(self.filter)

        self.framesKenBurnsButton = QPushButton("Frames", self)
        self.framesKenBurnsButton.clicked.connect(self.framesKenBurns)

        self.kenBurnsButton = QPushButton("Ken Burns", self)
        self.kenBurnsButton.clicked.connect(self.kenBurns)

    def _setupSliders(self):
        """
        Initialize UI sliders and connect them to their respective slots (functions).
        """
        self.penThicknessSlider = self._createSlider(
            Qt.Vertical, 1, 20, self.penThickness, self.updatePenThickness
        )
        self.penThicknessLabel = QLabel("Pen Thickness: 1", self)
        self.penThicknessLabel.setAlignment(Qt.AlignCenter)

        self.penColorSlider = self._createSlider(
            Qt.Vertical, 0, 255, 0, self.updatePenColor
        )
        self.penColorLabel = QLabel("Pen Color: 0", self)
        self.penColorLabel.setAlignment(Qt.AlignCenter)

        self.apertureSizeSlider = self._createSlider(
            Qt.Vertical, 28, 220, 28, self.updateApertureSize
        )
        self.apertureSizeLabel = QLabel("Aperture Size: f/2.8", self)
        self.apertureSizeLabel.setAlignment(Qt.AlignCenter)

    def _createSlider(
        self,
        orientation: Qt.Orientation,
        min_val: int,
        max_val: int,
        init_val: int,
        slot,
    ) -> QSlider:
        """
        Create and configure a QSlider, connecting it to a provided slot (function).

        Args:
            orientation (Qt.Orientation): Orientation of the slider (Vertical/Horizontal).
            min_val (int): Minimum value of the slider.
            max_val (int): Maximum value of the slider.
            init_val (int): Initial value of the slider.
            slot (Callable): Function to connect to the slider's valueChanged signal.

        Returns:
            QSlider: Configured slider.
        """
        slider = QSlider(orientation, self)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setFixedSize(140, 200)
        slider.valueChanged.connect(slot)
        return slider

    def _organizeLayouts(self):
        """
        Organize and set layouts for the widgets, arranging buttons, sliders, and labels.
        """
        depthMapButtonsLayout = QVBoxLayout()
        depthMapButtonsLayout.addWidget(self.depthMapButton)
        depthMapButtonsLayout.addWidget(self.depthMapCNNButton)

        kenBurnsButtonsLayout = QVBoxLayout()
        kenBurnsButtonsLayout.addWidget(self.framesKenBurnsButton)
        kenBurnsButtonsLayout.addWidget(self.kenBurnsButton)
        buttonLayout = QHBoxLayout()

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.loadImageButton)
        buttonLayout.addLayout(depthMapButtonsLayout)
        buttonLayout.addWidget(self.filterButton)
        buttonLayout.addLayout(kenBurnsButtonsLayout)

        sliderLayout = QVBoxLayout()
        sliderLayout.addStretch(1)
        for widget in [
            self.penThicknessLabel,
            self.penThicknessSlider,
            self.penColorLabel,
            self.penColorSlider,
            self.apertureSizeLabel,
            self.apertureSizeSlider,
        ]:
            sliderLayout.addWidget(widget)
        sliderLayout.addStretch(1)

        layout = QVBoxLayout()
        layout.addWidget(self.instructionLabel)
        layout.addWidget(self.imageLabel)
        layout.addLayout(buttonLayout)

        mainLayout = QHBoxLayout()
        mainLayout.addLayout(sliderLayout)
        mainLayout.addLayout(layout)

        central_widget = QWidget()
        central_widget.setLayout(mainLayout)
        self.setCentralWidget(central_widget)

    def updateInstruction(self):
        """
        Update the instruction label text based on the current application state.

        The instructions guide the user through the application workflow, providing
        context-sensitive guidance.
        """
        instructions = {
            "input": "You've loaded an image and now scribbles are active.<br>Draw on the image and then <b>generate the depth map using one of the methods available.</b>",
            "depthmap": "Depth map was generated by solving the Poisson's equation.<br>Select the focus depth on the image and the aperture size and note that: <b>f/2.8 -> very blurry ; f/22.0 -> not blurry at all</b>.<br>Then click on <b>filter</b> to generate the filtered image.",
            "depthmapCNN": "Depth map was generated by applying MiDaS CNN and combining its prediction with the scribbles.<br>Select the focus depth on the image and the aperture size and note that: <b>f/2.8 -> very blurry ; f/22.0 -> not blurry at all</b>.<br>Then click on <b>filter</b> to generate the filtered image.",
            "filter": "The filtered image was generated by applying a bilateral filter to the input image.<br>Click on <b>Frames</b> to choose the start and end frames for the Ken Burns effect.",
            "framesKenBurns": "Choose the start and end frames.<br>Click on <b>Ken Burns</b> to generate a video with the Ken Burns effects.",
            "kenBurns": "Here is the Ken Burns effect with depth-based parallax applied to the input image.<br>Click on <b>Load Image</b> to start over.",
        }
        self.instructionLabel.setText(instructions.get(self.currentState, ""))

    def handleStateChanged(self):
        """
        Handle state changes in the video player, looping the video.
        """
        if self.videoPlayer.state() == QMediaPlayer.StoppedState:
            self.videoPlayer.play()

    def updateImageDisplay(self, finalPixmap: QPixmap = None):
        """
        Scale and set the image to the display label.

        Args:
            finalPixmap (QPixmap, optional): The pixmap to display. If None, the current
                                            pixmap is used. Defaults to None.
        """
        if self.currentState == "kenBurns":
            self.videoOutput.setGeometry(self.imageLabel.rect())
            self.videoOutput.show()
            self.videoPlayer.setVideoOutput(self.videoOutput)
            self.videoPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(self.currentFilePath))
            )
            self.videoPlayer.stateChanged.connect(self.handleStateChanged)
            self.videoPlayer.play()
        else:
            # Hide ken burns video
            if self.videoPlayer.state() == QMediaPlayer.PlayingState:
                self.videoPlayer.stop()
            if self.videoOutput.isHidden() == False:
                self.videoOutput.hide()
            # If no pixmap is provided, use the current pixmap
            if finalPixmap is None:
                if (
                    self.currentState == "input"
                ):  # Combine image and scribbles if the current display is the input image
                    finalPixmap = self.currentPixmap.copy()
                    painter = QPainter(finalPixmap)
                    painter.drawPixmap(0, 0, self.scribblePixmap)
                    painter.end()
                else:
                    finalPixmap = self.currentPixmap

            # Scale the image while maintaining aspect ratio and display it
            finalPixmap_scaled = finalPixmap.scaled(
                self.imageLabel.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.imageLabel.setPixmap(finalPixmap_scaled)

    def loadImage(self, default_image: bool = False):
        """
        Load an image from a file path and display it.

        Args:
            default_image (bool, optional): If True, load a default image. If False, open a file
                                            ialog for the user to select an image. Defaults to False.
        """
        # Load input file
        if default_image:
            self.inputFilePath = self.args.file
        else:
            options = QFileDialog.Options()
            self.inputFilePath, _ = QFileDialog.getOpenFileName(
                self,
                "QFileDialog.getOpenFileName()",
                "",
                "All Files (*);;JPEG (*.jpg);;PNG (*.png)",
                options=options,
            )

        # If scribbles exist in the outputs folder, delete
        if os.path.exists("outputs/scribbles.png"):
            os.remove("outputs/scribbles.png")

        # Set current image as input and create annotations space
        if self.inputFilePath:
            self.currentFilePath = self.inputFilePath
            self.currentState = "input"  # Set current file as input
            self.currentPixmap = QPixmap(self.currentFilePath)  # Load image
            self.scribblePixmap = QPixmap(
                self.currentPixmap.size()
            )  # Initialize scribbleImage
            self.scribblePixmap.fill(Qt.transparent)  # Make it transparent
            self.updateInstruction()
            self.updateImageDisplay()

    def scribbleState(self):
        """
        Transition the application state to allow scribbling on the image.
        """
        if self.scribblePixmap and self.currentState == "input":
            # Change the current state to scribbles
            self.currentFilePath = "outputs/scribbles.png"
            self.currentState = "scribbles"
            self.currentPixmap = self.scribblePixmap

            # Save the scribbles on a transparent background
            output_folder = "outputs"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            self.currentPixmap.save(os.path.join(output_folder, f"scribbles.png"))

    def depthMap(self):
        """
        Generate and display a depth map based on user scribbles.
        """
        self.scribbleState()
        if self.currentState == "scribbles":
            # Change the current state to depthmap
            self.currentFilePath = "outputs/depthmap.png"
            self.currentState = "depthmap"

            # Create depth map
            dmg = DepthMapGenerator(beta=100.0, max_iter=7500)
            dmg.create_depth_map(
                self.inputFilePath, "outputs/scribbles.png", self.currentFilePath
            )
            self.currentPixmap = QPixmap(self.currentFilePath)

            # Load and Display the generated depth map
            self.updateInstruction()
            self.updateImageDisplay()

            # Change the current state
            self.currentState = "chooseFDepthAptSize"

    def depthMapCNN(self):
        """
        Generate and display a depth map using the MiDaS CNN model.
        """
        self.scribbleState()
        if self.currentState == "scribbles":
            # Change the current state to depthmapCNN
            self.currentFilePath = "outputs/depthmapCNN.png"
            self.currentState = "depthmapCNN"

            # Create depth map using MiDaS CNN
            dmcnng = DepthMapCNNGenerator(optimize_execution=self.args.optimize)
            dmcnng.create_depth_map_CNN(self.inputFilePath, self.currentFilePath)

            # If the scribbles are empty, use the depth map generated by CNN
            if (
                cv2.imread("outputs/scribbles.png", cv2.IMREAD_UNCHANGED)[:, :, 3] == 0
            ).all():
                cv2.imwrite("outputs/depthmap.png", cv2.imread(self.currentFilePath))
            else:
                self.currentFilePath = "outputs/depthmap.png"
                # Combine predicted depth map with user scribbles
                dmg = DepthMapGenerator(beta=100.0, max_iter=7500)
                dmg.create_depth_map(
                    "outputs/depthmapCNN.png",
                    "outputs/scribbles.png",
                    self.currentFilePath,
                )
            self.currentPixmap = QPixmap(self.currentFilePath)

            # Load and Display the generated depth map using CNN combined with user scribbles
            self.updateInstruction()
            self.updateImageDisplay()

            # Change the current state
            self.currentState = "chooseFDepthAptSize"

    def filter(self):
        """
        Generate and display a filtered image based on the generated depth map.
        """
        if self.currentState == "chooseFDepthAptSize" and self.focus_depth is not None:
            # Change the current state to filter
            self.currentFilePath = "outputs/filtered.png"
            self.currentState = "filter"

            # Create filtered image
            bf = BilateralFilter(
                kernel_size=7,
                focus_depth=self.focus_depth,
                aperture_size=self.apertureSize,
            )
            bf.apply_filter(
                self.inputFilePath, "outputs/depthmap.png", self.currentFilePath
            )
            self.currentPixmap = QPixmap(self.currentFilePath)

            # Load and Display the generated filtered image
            self.updateInstruction()
            self.updateImageDisplay()

    def framesKenBurns(self):
        """
        Generate and display frames for the Ken Burns effect.
        """
        if (
            self.currentState == "depthmap"
            or self.currentState == "depthmapCNN"
            or self.currentState == "chooseFDepthAptSize"
            or self.currentState == "filter"
            or self.currentState == "kenBurns"
        ):
            # Change the current state to framesKenBurns
            self.currentState = "framesKenBurns"
            self.currentPixmap = QPixmap(self.inputFilePath)

            # Display input image
            self.updateInstruction()
            self.updateImageDisplay()

    def kenBurns(self):
        """
        Generate and display a video with the Ken Burns effect.
        """
        if self.currentState == "framesKenBurns" and len(self.frames) == 2:
            # Change the current state to kenBurns
            self.currentState = "kenBurns"
            self.currentFilePath = "outputs/ken_burns.avi"  # "outputs/ken_burns.mp4" if mp4v coded was chosen
            self.currentFilePath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                self.currentFilePath,
            )
            self.updateInstruction()

            # Create Ken Burns effect
            kbe = KenBurnsEffect(
                self.inputFilePath,
                "outputs/depthmap.png",
                self.currentFilePath,
                self.frames,
                self.args.warping_factor,
                self.args.near_plane,
                self.args.far_plane,
                self.args.duration_video,
                self.args.anaglyph,
            )
            kbe.generate_video()

            # Display the video
            self.updateImageDisplay()

    def resizeEvent(self, event):
        """
        Rescale the image to fit the label upon window resizing.

        Args:
            event (QEvent): The event object containing details about the resize event.
        """
        if self.currentState == "kenBurns":
            self.videoOutput.setGeometry(self.imageLabel.rect())
        elif self.currentPixmap:
            self.updateImageDisplay()

    def getPixmapCoordinates(self, pos: QPoint) -> QPoint:
        """
        Convert QLabel coordinates to QPixmap coordinates.

        Args:
            pos (QPoint): Position in QLabel coordinates.

        Returns:
            QPoint: Position in QPixmap coordinates.
        """
        labelRect = self.imageLabel.contentsRect()
        pixmapSize = self.currentPixmap.size()
        scaledSize = pixmapSize.scaled(labelRect.size(), Qt.KeepAspectRatio)

        # Calculating the x and y offsets, considering keeping the image centered
        xOffset, yOffset = (labelRect.width() - scaledSize.width()) // 2, (
            labelRect.height() - scaledSize.height()
        ) // 2

        # Considering the width of the slider layout
        slidersWidth = self.penColorSlider.width()

        # Considering the height of the instruction label
        instructionLabelHeight = self.instructionLabel.height()

        # Considering layout margins and spacing
        mainLayout = self.centralWidget().layout()
        slidersWidth += (
            mainLayout.spacing()
            + mainLayout.contentsMargins().left()
            + mainLayout.contentsMargins().right()
        )
        instructionLabelHeight += (
            mainLayout.spacing()
            + mainLayout.contentsMargins().top()
            + mainLayout.contentsMargins().bottom()
        )

        # Convert QLabel coordinates to QPixmap coordinates
        # Adjust x considering the slider width, spacing, and margins
        x = int(
            (pos.x() - xOffset - labelRect.x() - slidersWidth)
            * (pixmapSize.width() / scaledSize.width())
        )
        y = int(
            (pos.y() - yOffset - labelRect.y() - instructionLabelHeight)
            * (pixmapSize.height() / scaledSize.height())
        )

        return QPoint(x, y)

    def mousePressEvent(self, event):
        """
        Handle mouse press events, initiating drawing or setting focus depth.

        Args:
            event (QMouseEvent): The event object containing details about the mouse press event.
        """
        # Check if the state is valid and the left mouse button was pressed
        if (
            not self.currentPixmap
            or (
                self.currentState != "input"
                and self.currentState != "chooseFDepthAptSize"
                and self.currentState != "framesKenBurns"
            )
            or event.button() != Qt.LeftButton
        ):
            return

        relativePos = self.getPixmapCoordinates(event.pos())
        imageRect = self.currentPixmap.rect()

        # Check if the mouse is inside the image
        if imageRect.contains(relativePos):
            if self.currentState == "input":
                self.drawing = True
                self.lastPoint = relativePos
                self.tempImage = (
                    self.currentPixmap.copy()
                )  # Create a temporary image to draw on
            elif (
                self.currentState == "chooseFDepthAptSize"
            ):  # Draw focus depth point (yellow)
                D = cv2.imread(self.currentFilePath, cv2.IMREAD_GRAYSCALE)
                focus_point_x, focus_point_y = relativePos.x(), relativePos.y()
                self.focus_depth = D[focus_point_y, focus_point_x]

                # Use the tempImage to draw, so previously drawn points aren’t included
                self.tempImage = self.currentPixmap.copy()
                painter = QPainter(self.tempImage)
                pen = QPen(Qt.yellow, 5, Qt.SolidLine)  # Adjust the size accordingly
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                radius = 2  # radius of the circle
                painter.setBrush(Qt.yellow)
                painter.drawEllipse(relativePos, radius, radius)
                painter.end()

                # Update displayed image
                self.updateImageDisplay(self.tempImage)
            elif self.currentState == "framesKenBurns":
                if len(self.frames) == 2:
                    self.frames.clear()
                    self.updateImageDisplay(
                        self.currentPixmap
                    )  # Reset image to remove old frames
                self.tempImage = self.currentPixmap.copy()
                self.drawing = True
                self.startFramePoint = relativePos
                self.endFramePoint = relativePos

    def mouseMoveEvent(self, event):
        """
        Handle mouse move events, drawing on the image if in drawing mode.

        Args:
            event (QMouseEvent): The event object containing details about the mouse move event.
        """
        if (
            not self.currentPixmap
            or (self.currentState != "input" and self.currentState != "framesKenBurns")
            or not self.drawing
        ):
            return

        relativePos = self.getPixmapCoordinates(event.pos())
        imageRect = self.currentPixmap.rect()

        if imageRect.contains(relativePos):
            if self.currentState == "input":
                painter = QPainter(self.scribblePixmap)  # Draw on scribbleImage
                pen = QPen(self.penColor, self.penThickness, Qt.SolidLine)
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.drawLine(self.lastPoint, relativePos)
                painter.end()

                self.lastPoint = relativePos
                self.updateImageDisplay()  # Update to reflect the new scribble
            elif self.currentState == "framesKenBurns" and self.drawing:
                aspect_ratio = self.currentPixmap.width() / self.currentPixmap.height()

                intended_end_x = relativePos.x()
                intended_end_y = relativePos.y()

                intended_width = intended_end_x - self.startFramePoint.x()
                intended_height = intended_end_y - self.startFramePoint.y()

                # Calculate width and height maintaining the aspect ratio
                if intended_width / aspect_ratio <= intended_height:
                    height = intended_width / aspect_ratio
                    width = intended_width
                else:
                    width = intended_height * aspect_ratio
                    height = intended_height

                self.endFramePoint = QPoint(
                    int(self.startFramePoint.x() + width),
                    int(self.startFramePoint.y() + height),
                )

                self.tempImage = self.currentPixmap.copy()  # Start with a clean image
                painter = QPainter(self.tempImage)
                if len(self.frames) == 0:
                    painter.setPen(
                        QPen(QColor(46, 113, 183), 2, Qt.SolidLine)
                    )  # Blue for the start frame
                else:
                    painter.setPen(
                        QPen(QColor(46, 113, 183), 2, Qt.SolidLine)
                    )  # Blue for the start frame
                    painter.drawRect(self.frames[0])  # Draw the start frame
                    painter.setPen(
                        QPen(QColor(201, 91, 46), 2, Qt.SolidLine)
                    )  # Orange for the end frame
                painter.drawRect(
                    QRect(self.startFramePoint, self.endFramePoint).normalized()
                )
                painter.end()
                self.updateImageDisplay(self.tempImage)

    def mouseReleaseEvent(self, event):
        """
        Handle mouse release events, finalizing drawing if in drawing mode.

        Args:
            event (QMouseEvent): The event object containing details about the mouse release event.
        """
        if (
            not self.currentPixmap
            or (self.currentState != "input" and self.currentState != "framesKenBurns")
            or event.button() != Qt.LeftButton
            or not self.drawing
        ):
            return

        if self.currentState == "input":
            self.drawing = False
            self.currentPixmap = self.tempImage.copy()
        elif self.currentState == "framesKenBurns":
            self.drawing = False
            frameRect = QRect(self.startFramePoint, self.endFramePoint).normalized()
            self.frames.append(frameRect)

            # Drawing the frame
            self.tempImage = self.currentPixmap.copy()
            painter = QPainter(self.tempImage)
            if len(self.frames) == 1:
                painter.setPen(
                    QPen(QColor(46, 113, 183), 2, Qt.SolidLine)
                )  # Blue for the start frame
            else:
                painter.setPen(
                    QPen(QColor(46, 113, 183), 2, Qt.SolidLine)
                )  # Blue for the start frame
                painter.drawRect(self.frames[0])  # Draw the start frame
                painter.setPen(
                    QPen(QColor(201, 91, 46), 2, Qt.SolidLine)
                )  # Orange for the end frame
            painter.drawRect(frameRect)
            painter.end()
            self.updateImageDisplay(self.tempImage)  # Show frames on the image

    def updatePenThickness(self):
        """
        Update the pen thickness based on the slider value.
        """
        self.penThickness = self.penThicknessSlider.value()
        self.penThicknessLabel.setText(
            f"Pen Thickness: {self.penThicknessSlider.value()}"
        )

    def updatePenColor(self):
        """
        Update the pen color based on the slider value.
        """
        intensity = self.penColorSlider.value()
        self.penColor = QColor(intensity, intensity, intensity)
        self.penColorLabel.setText(f"Pen Color: {self.penColorSlider.value()}")

    def updateApertureSize(self):
        """
        Update the aperture size based on the slider value.
        """
        self.apertureSize = self.apertureSizeSlider.value() / 10
        self.apertureSizeLabel.setText(f"Aperture Size: f/{self.apertureSize}")
