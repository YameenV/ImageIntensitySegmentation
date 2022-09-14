from utils import loadDataFrame, loadImage, imgToDataList, quantiletransformDataFrame

PATH = "./test6.jpg"
img = loadImage(PATH)
imgList = imgToDataList(img[1], img[2], img[0])
df = loadDataFrame(imgList)
dfnew = quantiletransformDataFrame(df)

#Parameter
left = df["quantile"].mean() + -0.5* df["quantile"].std()
right = df["quantile"].mean() + 1* df["quantile"].std()


# Lopping over the image an aloting a value to a pixel
tempImg = np.zeros((img[1],img[2], 3),img[0].dtype)
for val, quan in zip(df["position"], df["quantile"]):
  x = val[1]
  y = val[0]
  # loop over all the three channel of the Image ->> Red Green Blue
  if left >= quan:
    values = [0,51,102]
    for i in range(3):
      tempImg[y,x,i] = values[i]
  if left < quan and right > quan:
    values = [51,153,255]
    for i in range(3):
      tempImg[y,x,i] = values[i]
  if right <=  quan:
    values = [204,229,255]
    for i in range(3):
      tempImg[y,x,i] = values[i]

#Transform Image
tempImg