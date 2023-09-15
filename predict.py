import os
import torch
from PIL import Image
from train import NeuralNetWork
from loader import one_hot_decode
from torchvision import transforms


def predict(model, file_path):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    with torch.no_grad():
        X = trans(Image.open(file_path)).reshape(1, 1, 60, 160)
        pred = model(X)
        text = one_hot_decode(pred)
        return text


def main():
    model = NeuralNetWork()
    model.load_state_dict(torch.load("./model.pth", map_location=torch.device("cpu")))
    model.eval()

    correct = 0
    test_dir = "./test_captcha"
    total = len(os.listdir(test_dir))
    for filename in os.listdir(test_dir):
        file_path = f"{test_dir}/{filename}"
        real_captcha = file_path.split("-")[-1].replace(".png", "")
        pred_captcha = predict(model, file_path)

        if pred_captcha == real_captcha:
            correct += 1
            print(f"{file_path}的预测结果为{pred_captcha}，预测正确")
        else:
            print(f"{file_path}的预测结果为{pred_captcha}，预测错误")

    accuracy = f"{correct / total * 100:.2f}%"
    print(accuracy)


main()