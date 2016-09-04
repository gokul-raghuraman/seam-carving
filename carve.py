import cv2
import numpy as np

class Carver:

	def __init__(self, imgPath, scale=1):
		img = cv2.imread(imgPath)
		scaledWidth = int(img.shape[1] * scale)
		scaledHeight = int(img.shape[0] * scale)
		self.img = cv2.resize(img, (scaledWidth, scaledHeight), interpolation=cv2.INTER_CUBIC)
		self.imgWidth = self.img.shape[1]
		self.imgHeight = self.img.shape[0]

	def shrinkX(self, numPixels):
		energyImage = np.zeros([self.imgHeight, self.imgWidth, 3], dtype=np.uint8)
		for row in range(self.imgHeight):
			for col in range(self.imgWidth):
				e1Error = self._getE1Error(row, col)
				energyImage[row, col, :] = [e1Error, e1Error, e1Error]
		
		seamColIdxs = self.computeOptimalVerticalSeam()

		# TEMP: draw seam
		imgTemp = np.copy(self.img)
		for row in range(self.imgHeight):
			imgTemp[row, seamColIdxs[row]] = [0, 0, 255]
		cv2.imshow("With Seam", imgTemp)
		cv2.waitKey(0)
		# TEMP: draw seam


		# Shrink the image along the optimal seam
		for row in range(self.imgHeight):
			colIdx = seamColIdxs[row]
			for col in range(colIdx, self.imgWidth-1):
				self.img[row, col, :] = self.img[row, col+1, :]
		self.img = self.img[:, :self.imgWidth-1, :]
		self.imgWidth -= 1

	def colorEnergyImageSimple(self, rawEnergyImage):
		coloredImage = np.zeros([rawEnergyImage.shape[0], rawEnergyImage.shape[1], 3], dtype=np.uint8)
		for row in range(rawEnergyImage.shape[0]):
			for col in range(rawEnergyImage.shape[1]):
				grayLevel = rawEnergyImage[row, col, 0]

				# OpenCV uses BGR color order
				if grayLevel < 85:
					coloredImage[row, col, :] = [255, 0, 0]
				elif grayLevel < 170:
					coloredImage[row, col, :] = [0, 255, 0]
				else:
					coloredImage[row, col, :] = [0, 0, 255]
				
		return coloredImage
	
	def computeOptimalVerticalSeam(self):
		energyMatrix = self._getEnergyMatrix()

		optimalEnergyMatrix = np.zeros([self.imgHeight, self.imgWidth], dtype=np.uint8)
		#print(energyMatrix.shape)
		#print(optimalEnergyMatrix.shape)
		optimalEnergyMatrix[0, :] = energyMatrix[0, :]

		for row in range(1, self.imgHeight):
			for col in range(self.imgWidth):
				prevColPixVal = 0
				if not col == 0:
					prevColPixVal = optimalEnergyMatrix[row-1, col-1]
				nextColPixVal = 0
				if not col == self.imgWidth-1:
					nextColPixVal = optimalEnergyMatrix[row-1, col+1]
				optimalEnergyMatrix[row, col] = energyMatrix[row, col] + \
					np.minimum(np.minimum(np.int64(prevColPixVal), np.int64(nextColPixVal)), np.int64(optimalEnergyMatrix[row-1, col]))

		# backtrack
		seamColIdxs = np.zeros(self.imgHeight, dtype=np.int64)
		startCol = np.argmin(optimalEnergyMatrix[self.imgHeight-1, :])
		seamColIdxs[self.imgHeight-1] = startCol
		curCol = startCol
		for row in range(self.imgHeight-2, -1, -1):
			# Need to add seamColIdxs[row+1] - 1 as in below, because np.argmin returns the idx
			# relative to the passed-in array. So we'll get a value between 0-2
			optimalColIdx = (seamColIdxs[row+1] - 1) + np.argmin(optimalEnergyMatrix[row, seamColIdxs[row+1]-1:seamColIdxs[row+1]+2])
			seamColIdxs[row] = optimalColIdx

		return seamColIdxs

	def _getEnergyMatrix(self):
		energyImage = np.zeros([self.imgHeight, self.imgWidth], dtype=np.uint8)
		for row in range(self.imgHeight):
			for col in range(self.imgWidth):
				e1Error = self._getE1Error(row, col)
				energyImage[row, col] = e1Error
		return energyImage

	def _getE1Error(self, row, col):
		e1Error = 0.0
		for c in range(3):
			curPixVal = self.img[row, col, c]
			prevColPixVal = curPixVal
			nextColPixVal = curPixVal
			prevRowPixVal = curPixVal
			nextRowPixVal = curPixVal

			if row == 0:
				prevRowPixVal = 0
			else:
				prevRowPixVal = self.img[row-1, col, c]
			if row == self.img.shape[0] - 1:
				nextRowPixVal = 0
			else:
				nextRowPixVal = self.img[row+1, col, c]

			if col == 0:
				prevColPixVal = 0
			else:
				prevColPixVal = self.img[row, col-1, c]
			if col == self.img.shape[1] - 1:
				nextColPixVal = 0
			else:
				nextColPixVal = self.img[row, col+1, c]

			e1Error += np.abs(np.int64(prevColPixVal) - np.int64(nextColPixVal)) + np.abs(np.int64(prevRowPixVal) - np.int64(nextRowPixVal))
		return e1Error



if __name__ == "__main__":
	imgPath = "./img3.png"
	carver = Carver(imgPath, 0.5)
	for i in range(20):
		carver.shrinkX(1)
		print(i+1)
	
	cv2.imshow("carved", carver.img)
	cv2.waitKey(0)

