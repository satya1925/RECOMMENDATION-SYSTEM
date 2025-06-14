
# RECOMMENDATION SYSTEM

This project demonstrates the implementation of a Collaborative Filtering-based Movie Recommendation System using Matrix Factorization with the SVD algorithm. It uses the MovieLens 100k dataset and evaluates the system using standard metrics.

## Internship Information

- **Company**: CODTECH IT SOLUTIONS PVT. LTD  
- **Name**: Peethani Satya Durga Rao  
- **Intern ID**: CT06DF395  
- **Domain**: Machine Learning  
- **Duration**: 6 Weeks  
- **Mentor**: Neela Santhosh Kumar  

## Task Description

The goal of this task is to build a recommendation system that predicts how users would rate movies and provides personalized movie suggestions. We utilize the `Surprise` library and the SVD (Singular Value Decomposition) algorithm for collaborative filtering.

### Objectives

- Understand collaborative filtering using matrix factorization  
- Train an SVD model on MovieLens 100k dataset  
- Evaluate the model using RMSE and MAE  
- Generate personalized movie recommendations  
- Visualize prediction errors  

---

## Steps

### 1. Importing Required Libraries

```python
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
````

---

### 2. Load and Prepare Dataset

```python
data = Dataset.load_builtin('ml-100k')
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
print("RMSE:", rmse(predictions))
print("MAE:", mae(predictions))
```

> Example Output:

```
RMSE: 0.93
MAE: 0.73
```

---

### 5. Generate Top-N Movie Recommendations

```python
def get_top_n(predictions, n=5):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)
```

---

### 6. Display Recommendations for Sample Users

```python
for uid, user_ratings in list(top_n.items())[:3]:
    print(f"User {uid}:")
    for (iid, rating) in user_ratings:
        print(f"  Movie ID: {iid} | Predicted Rating: {rating:.2f}")
```

---

### 7. Visualize Prediction Errors

```python
import matplotlib.pyplot as plt

errors = [abs(true_r - est) for (_, _, true_r, est, _) in predictions]
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Prediction Error Distribution")
plt.xlabel("Absolute Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
```

---

### 8. Show Recommendations with Movie Titles

```python
item_df = pd.read_csv(
    'https://files.grouplens.org/datasets/movielens/ml-100k/u.item',
    sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['movie_id', 'title']
)
movie_map = dict(zip(item_df.movie_id.astype(str), item_df.title))

uid = list(top_n.keys())[0]
for iid, rating in top_n[uid]:
    print(f"{movie_map.get(iid, 'Unknown')} - Predicted Rating: {rating:.2f}")
```

> Example Output:

```
Star Wars (1977) - Predicted Rating: 4.88
Shawshank Redemption, The (1994) - Predicted Rating: 4.85
```

---

## Final Summary

* Matrix factorization with SVD provided effective collaborative filtering
* RMSE and MAE scores were under 1.0, showing strong predictive performance
* Personalized top-N recommendations were generated for users

---

**📁 NOTE**: This project was implemented in a Jupyter Notebook, and includes all code blocks, outputs, and evaluation visualizations.

```

Let me know if you’d like this exported into a `.zip` with the notebook and README bundled together.
```
