# tensorPrice
Automated Receipt Processing
---

This is my attempt to retrain TensorFlow Image Classifier. I wanted to make a automaticly process recipts and export main information from them (Retailer, Total Price). Retail recognision works at about 70% accurancy and at this moment can recognise receipts from Maxima and Lidl supermarkets.

I want to increase retailer detection accurancy but it will take some time to collect more data to train my model on.

Processing example:

![](https://i.imgur.com/85Sl3QA.jpg)

Known issues:
* Low retailer detection accurancy
* OCR detects and exports the biggest number it finds in the receipt ignoring other valuable information.

[Dataset](https://www.dropbox.com/sh/s2c8tpwu0hzpmzn/AAA0tIPwcMb4Swk1uNi511lIa?dl=0)

Built with:
* [React](https://reactjs.org/)
* [Golang](https://golang.org/)
* [TensorFlow](https://www.tensorflow.org/)
