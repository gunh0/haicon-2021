# HAICon2021 (HIL-based Augmented ICS)

> https://dacon.io/en/competitions/open/235757/overview/description

### Reference

-   HAI 21.03, HAICon 2020
    -   Hyeok-ki Shin, Woomyo Lee, Jeong-Han Yun, and Byung-Gil Min, "Two ICS Security Datasets and Anomaly Detection Contest on the HIL-based Augmented ICS Testbed", CSET'21: Workshop on Cyber Security Experimentation and Test, 2021
-   HAI 20.07
    -   Hyeok-ki Shin, Woomyo Lee, Jeong-Han Yun, and Hyoungchun Kim, "HAI 1.0: HIL-based Augmented ICS Security Dataset", CSET'20: Workshop on Cyber Security Experimentation and Test, 2020.
-   TaPR
    -   Won-seok Hwang, Jeong-Han Yun, Jonguk Kim, and Hyoungchun Kim, "Time-Series Aware Precision and Recall for Anomaly Detection - Considering Variety of Detection Result and Addressing Ambiguous Labeling", CIKM'19: Proceedings of the 28th ACM International Conference on Information and Knowledge Management, 2019.

<br/>

### Install eTaPR

```bash
pip install eTaPR-21.8.2-py3-none-any.whl
```

<br/>

### Download dataset

-   https://drive.google.com/file/d/1jzLP4uPWo71erTwx0K9OECZpbmrimzYG/view

<br/>

### 2020 Result Table

| score   | window      | hidden | layer     | batch | stride | epoch | threshold             |
| ------- | ----------- | ------ | --------- | ----- | ------ | ----- | --------------------- |
| 0.93793 |             |        |           |       |        |       |                       |
| 0.93614 | 59+1 / 9+10 |        | LSTM \* 3 | 512   |        | 32    | 0.027 / 0.019 / 0.008 |
| 0.92752 |             |        |           |       |        |       |                       |
| 0.93414 |             |        |           |       |        |       |                       |
| 0.92365 |             |        |           |       |        |       |                       |

<br/>

### Result Table

| score  | F1    | window | hidden | layer | batch | stride | epoch |
| ------ | ----- | ------ | ------ | ----- | ----- | ------ | ----- |
| 0.2764 | 0.172 | 100    | 82     | 3     | 1536  | 4      | 70    |
| 0.2802 | 0.131 | 89+1   | 50     | 3     | 216   | 10     |       |
