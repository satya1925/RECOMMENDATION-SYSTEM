
# RECOMMENDATION SYSTEM

This project demonstrates the implementation of a **Collaborative Filtering-based General-Purpose Recommendation System** using **Matrix Factorization with the SVD algorithm**. It uses a simulated dataset of user-item interactions and evaluates the system using standard metrics.

## Internship Information

- **Company**: CODTECH IT SOLUTIONS PVT. LTD  
- **Name**: Peethani Satya Durga Rao  
- **Intern ID**: CT06DF395  
- **Domain**: Machine Learning  
- **Duration**: 6 Weeks  
- **Mentor**: Neela Santhosh Kumar  

## Task Description

The goal of this task is to build a recommendation system that predicts how users would interact with different items and provides personalized suggestions. We utilize the `Surprise` library and the SVD (Singular Value Decomposition) algorithm for collaborative filtering.

### Objectives

- Understand collaborative filtering using matrix factorization  
- Train an SVD model on custom user-item rating data  
- Evaluate the model using RMSE and MAE  
- Generate personalized top-N item recommendations  
- Analyze prediction performance  

---

## Steps

### 1. Importing Required Libraries

```python
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
```

---

### 2. Load and Prepare Dataset

```python
data_dict = {
    'user_id': ['u1', 'u2', 'u1', 'u3', 'u2', 'u3', 'u1'],
    'item_id': ['i1', 'i1', 'i2', 'i2', 'i3', 'i3', 'i3'],
    'rating':  [4, 5, 2, 3, 5, 4, 3]
}
df = pd.DataFrame(data_dict)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
```

---

### 3. Train the SVD Model

```python
model = SVD()
model.fit(trainset)
```

---

### 4. Evaluate Model Performance

```python
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))
print("MAE:", accuracy.mae(predictions))
```

> Example Output:

```
RMSE: 0.82
MAE: 0.65
```

---

### 5. Generate Top-N Recommendations

```python
def get_top_n(predictions, n=3):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=3)
```

---

### 6. Display Recommendations for Sample Users

```python
for uid, user_ratings in top_n.items():
    print(f"User {uid}:")
    for iid, rating in user_ratings:
        print(f"  Item ID: {iid} | Predicted Rating: {rating:.2f}")
```

> Example Output:

```
User u1:
  Item ID: i3 | Predicted Rating: 3.72
  Item ID: i2 | Predicted Rating: 2.98
```

---

## Final Summary

* Matrix factorization with SVD enabled personalized item recommendations.  
* RMSE and MAE metrics showed acceptable prediction accuracy.  
* Top-N predictions can be used for any type of content (products, courses, books, etc).  
