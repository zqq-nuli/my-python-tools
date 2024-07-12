
import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\xxx\Tesseract-OCR\tesseract.exe"
)


def getNumbp(filePath):
    # 读取图像
    image = cv2.imread(filePath)

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Pytesseract进行OCR
    custom_config = r"--oem 3 --psm 6 outputbase digits"
    text = pytesseract.image_to_string(gray, config=custom_config)
    return str(text).strip()


if __name__ == "__main__":
    text = getNumbp("./iamges/1.jpg")
    print(text)
    text = getNumbp("./iamges/2.jpg")
    print(text)
    text = getNumbp("./iamges/3.jpg")
    print(text)
    text = getNumbp("./iamges/4.jpg")
    print(text)
    text = getNumbp("./iamges/5.jpg")
    print(text)
