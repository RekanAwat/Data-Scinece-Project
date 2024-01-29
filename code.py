import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sb
import pandas as pd
import re
from datetime import datetime

data = pd.read_csv('GraphicDesign.csv')
data_shape = data.shape
print("\nOriginal dataset shape: ", data_shape, "\n")

# detect and removing nulls
number_of_null = data.isnull().sum()
print("number of nulls:\n", number_of_null)

remove_null = data.dropna(inplace=True)
print("\ndataset shape after removing nulls: ", data.shape)

# detect and removing duplicates
number_of_duplicate = data.duplicated().sum()
print("\nnumber of duplicates:", number_of_duplicate)
remove_duplicate = data.drop_duplicates(inplace=True)
print("dataset shape after removing duplicates: ", data.shape)

# detect and fix wrong format values from Column Price
print(data.loc[data['price'] == 'Free', 'price'])
data['price'] = pd.to_numeric(data['price'], errors='coerce').astype('Int64').fillna(0)
print(data.loc[data['price'] == 0, 'price'])

# detect how many courses are free or paid
isPaid = data['isPaid']

counts = isPaid.value_counts()
color = ['violet', 'teal']
counts.plot(kind='bar', color=color)
plt.xlabel('Is Paid')
plt.ylabel('Frequency')
plt.title('Is Paid bar Chart')
plt.xticks(rotation=0)
plt.show()

paid = isPaid.sum()  # it's count only values that are True
free = len(isPaid) - paid
paid_free = [paid, free]
labels = ['Paid', 'Free']

plt.pie(paid_free, labels=labels, colors=color, autopct='%1.2f%%', startangle=120, explode=(0.1, 0.0))
plt.legend()
plt.show()

# describing price column like min, max, mean, etc
print(data['price'].describe())
price = data['price']

plt.hist(price, color="purple", alpha=0.6, rwidth=0.9, bins=15)
plt.xticks(range(0, price.max() + 1, 20))
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.show()

ranges = [-0.01, 1, 25, 75, 150, data['price'].max()]
counts = pd.cut(price, bins=ranges).value_counts().sort_index()
color = ['skyblue', 'yellowgreen', 'lightcoral', 'lightsalmon', 'lightpink']

counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Prices')
plt.show()

plt.pie(counts, autopct='%1.2f%%', colors=color)
plt.title('Price Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.85, 1))
plt.show()


sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sb.histplot(price, bins=ranges, color='green', kde=True)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Price')
plt.show()

# describing numSubscribers column like min, max, mean, etc

subscriber = data['numSubscribers']
print(subscriber.describe())

ranges = [-0.01, 100, 500, 1500, 5000, subscriber.max()]
counts = pd.cut(subscriber, bins=ranges).value_counts().sort_index()
color = ['lightseagreen', 'mediumpurple', 'deepskyblue', 'orchid', 'springgreen', 'tomato']

counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Number of Subscribers')
plt.show()

plt.pie(counts, autopct='%1.2f%%', colors=color)
plt.title('Subscriber Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.85, 1))
plt.show()

sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sb.histplot(subscriber, bins=ranges, color='green', kde=True)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Price')
plt.show()


filtered_subscribers = subscriber[(subscriber >= 0) & (subscriber <= 5000)]
print(filtered_subscribers.describe().round(2))

plt.hist(filtered_subscribers, color="coral", bins=10)
plt.xticks(range(0, 5010, 500))
plt.xlabel('Subscribers')
plt.ylabel('Frequency')
plt.title('Distribution of Subscribers (0-5000)')
plt.show()

ranges = [-0.01, 100, 500, 1500, 5000, filtered_subscribers.max()]
counts = pd.cut(filtered_subscribers, bins=ranges, duplicates='drop').value_counts().sort_index()
color = ['lightseagreen', 'mediumpurple', 'deepskyblue', 'orchid', 'springgreen', 'tomato']

counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Number of Subscribers')
plt.title('Subscriber Ranges (0-5000)')
plt.show()

plt.pie(counts, autopct='%1.2f%%', colors=color)
plt.title('Subscriber Ranges (0-5000)')
plt.legend(counts.index, bbox_to_anchor=(0.85, 1))
plt.show()

sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sb.histplot(filtered_subscribers, bins=ranges, color='green', kde=True)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Subscribers (0-5000)')
plt.show()





# column review
review = data['numReviews']
print(review.describe().round(2))

plt.hist(review, bins=15, color='teal')
plt.xticks(range(0, int(review.max()) + 1, 200))
plt.xlabel('reviews')
plt.ylabel('Frequency')
plt.title('Distribution of reviews')
plt.show()

ranges = [0, 9, 20, 40, 60, 100, 250, 750, max(review)]
counts = pd.cut(review, bins=ranges).value_counts().sort_index()
color = ['darkorchid', 'tomato', 'darkorange', 'gold', 'limegreen', 'mediumseagreen', 'turquoise', 'deepskyblue']

counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Number of Reviews')
plt.show()

plt.pie(counts, autopct='%1.2f%%', colors=color)
plt.title('Review Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.9, 1.1))
plt.show()

# corr between number of subscriber and number of reviews

color = ['dodgerblue' if r == review.max() else 'rebeccapurple' if s == subscriber.max() else 'hotpink' for r, s in
         zip(review, subscriber)]

plt.scatter(subscriber, review, color=color)
plt.xticks(range(0, int(subscriber.max()), 4000))
max_review_index = review.idxmax()
max_subscriber_index = subscriber.idxmax()

plt.text(subscriber[max_review_index] + 30, review[max_review_index] + 30,
         (subscriber[max_review_index], review[max_review_index]), color='dodgerblue')

plt.text(subscriber[max_subscriber_index] - 50, review[max_subscriber_index] - 30,
         (subscriber[max_subscriber_index], review[max_subscriber_index]), color='rebeccapurple', ha='right', va='top')

legend_labels = {'dodgerblue': 'Max Review', 'rebeccapurple': 'Max Subscriber', 'hotpink': 'Other'}
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=k, markersize=10, label=v) for k, v in
                   legend_labels.items()]
plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1))
plt.show()

# number of lectures

lecture = data['numPublishedLectures']
print(lecture.describe().round(2).astype('int64'))

plt.figure(figsize=(12, 6))
plt.hist(lecture, bins=40, color='violet', rwidth=0.9)
plt.xticks(range(0, int(lecture.max())+ 10, 10))
plt.xlabel("number of lectures")
plt.ylabel('frequency')
plt.title('Distribution of number of Lectures')
plt.show()

ranges = [0, 10, 20, 35 , 60, 100, max(lecture)]
counts = pd.cut(lecture, bins=ranges).value_counts().sort_index()
color = ['Red', 'salmon', 'orange', 'firebrick', 'yellow', 'mediumspringgreen']

plt.figure(figsize=(10 , 5))
counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Number of Lectures')
plt.title('Distribution of number of Lectures')
plt.show()

plt.figure(figsize=(10 , 5))
plt.pie(counts, autopct='%1.2f%%', colors=color, startangle = 180)
plt.title('Review Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.9, 1.1))
plt.show()


sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sb.histplot(lecture, bins=ranges, color='red', kde=True)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Lectures')
plt.show()

# content info


def convert_to_hours(duration):
    match = re.match(r'(\d+(\.\d+)?)\s*(\w*)', duration)
    if match:
        value, _, unit = match.groups()
        value = float(value)
        # Convert minutes to hours
        if unit in ['min', 'mins', 'minute', 'minutes']:
            return value / 60
        elif unit in ['hour', 'hours']:
            return value
        else:
            return value  # If no unit specified, assume it's already in hours
    else:
        return None  # Unable to match the pattern

# Apply the conversion function to the 'duration' column
data['contentInfo'] = data['contentInfo'].apply(convert_to_hours).round(2)

content = data['contentInfo']

# content info
print(content.describe().round(2))
plt.figure(figsize=(12, 6))
plt.hist(content, bins=40, color='violet', rwidth=0.9)
plt.xticks(range(0, int(content.max()), 4))
plt.xlabel("content hours")
plt.ylabel('frequency')
plt.title('Distribution of number of content hours')
plt.show()

ranges = [0, 1, 4, 10 , 25 ,max(content)]
counts = pd.cut(content, bins=ranges).value_counts().sort_index()
color = ['teal', 'deepskyblue', 'darkviolet', 'royalblue', 'navy' , 'slateblue']

plt.figure(figsize=(10 , 5))
counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('content hours')
plt.title('Distribution of Content Hours')
plt.show()

plt.figure(figsize=(10 , 5))
plt.pie(counts, autopct='%1.2f%%', colors=color, startangle = 0)
plt.title('Content Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.9, 1.1))
plt.show()


sb.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sb.histplot(content, bins=ranges, color='red', kde=True)


plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Content info')
plt.show()


# level
level = data['instructionalLevel']
level_counts = level.value_counts()

ranges = ['Beginner', 'All Levels', 'Intermediate Level', 'Expert Level']
color = ['Red', 'salmon', 'orange', 'springgreen']

plt.figure(figsize=(10, 5))
plt.bar(ranges, level_counts, color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Level of Courses')
plt.title('Distribution of Level of Courses')
plt.show()

plt.figure(figsize=(10, 5))
plt.pie(level_counts, labels=ranges, autopct='%1.2f%%', colors=color, startangle=180)
plt.title('Distribution of Level of Courses')
plt.show()

# published time

def convert_to_custom_format(date_time_string):
    date_time_object = datetime.strptime(date_time_string, "%Y-%m-%dT%H:%M:%SZ")
    return date_time_object.strftime("%Y/%m/%d")


# Apply the function to the 'timestamp' column
data['publishedTime'] = data['publishedTime'].apply(convert_to_custom_format)
publishedTime = data['publishedTime']

print(publishedTime.describe().round(2))

df = pd.DataFrame(publishedTime)

# Convert 'publishedTime' to datetime format
df['publishedTime'] = pd.to_datetime(df['publishedTime'])

# Extract year from 'publishedTime'
df['year'] = df['publishedTime'].dt.year

# Get the count of occurrences for each unique year
year_counts = df['year'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(10, 6))
color = ['deeppink', 'darkblue',  'royalblue', 'mediumorchid', 'rebeccapurple', 'mediumblue']
plt.bar(year_counts.index, year_counts.values, color=color)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Bar Chart of Published Time (Yearly)')
plt.show()

plt.figure(figsize=(10, 5))
plt.pie(year_counts, labels = year_counts.index,  autopct='%1.2f%%',  textprops={'color': 'white'}, colors=color, startangle=180)
plt.title('Distribution of Published Date')
plt.legend(bbox_to_anchor = (1,1))
plt.show()

# rating

rating = data['rating']

print(rating.describe().round(2))

ranges = [4.99 , 5 , 6 , 7 , 8 , 9]
counts = pd.cut(rating, bins=ranges).value_counts().sort_index()
color = ['darkorchid', 'darkorange', 'limegreen', 'mediumseagreen', 'turquoise']

counts.plot(kind='bar', color=color)
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.xlabel('Number of rating')
plt.show()

plt.pie(counts, autopct='%1.2f%%', colors=color)
plt.title('rating Ranges')
plt.legend(counts.index, bbox_to_anchor=(0.9, 1.1))
plt.show()

# correlation between number of subscriber (5000) and rating
filtered_subscribers = subscriber[(subscriber >= 0) & (subscriber <= 5000)]
filtered_rating = rating[(subscriber >= 0) & (subscriber <= 5000)]

sb.set(style="darkgrid")

plt.figure(figsize=(10, 6))
sb.scatterplot(x= filtered_rating, y=filtered_subscribers)
plt.yticks(range(0, int(filtered_subscribers.max()) + 1, 500))
plt.xlabel('Rating')
plt.ylabel('Subscribers')
plt.title('Rating & Subscribers for Subscribers below 5000')
plt.show()

# correlation between number of Reviews and rating
sb.set(style="darkgrid")

plt.figure(figsize=(10, 6))
sb.scatterplot(x= rating, y=review)
plt.yticks(range(0, int(review.max()) + 1, 300))
plt.xlabel('Rating')
plt.ylabel('Reviews')
plt.title('Rating & Reviews')
plt.show()

sb.set(style="darkgrid")

plt.figure(figsize=(10, 6))
sb.scatterplot(x= rating, y=price)
plt.xlabel('Rating')
plt.ylabel('Price')
plt.title('Rating & Price')
plt.show()

# correlation between price and level

sb.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sb.scatterplot(x= price, y=level)
plt.xlabel('Price')
plt.ylabel('Level')
plt.title('Scatter Plot between Price and Level')
plt.show()

#

plt.figure(figsize=(8, 6))
sb.barplot(x=level, y= subscriber,estimator=sum)

plt.xlabel('Level')
plt.yticks(range(0 , 1000000 , 100000))
plt.ylabel('Sum of Subscriber')
plt.title('Sum of Subscriber for Each Level')
plt.show()

# correlation between year and content
plt.figure(figsize=(8, 6))
sb.barplot(x=year_counts.index, y=year_counts, palette='viridis' , hue=year_counts.index)

plt.xlabel('Year')
plt.ylabel('Count of Hours of content')
plt.title('Number of hours of Content for Each Year')
plt.show()



# Correlation between numeric columns
corr = data.corr(numeric_only=True)
sb.heatmap(corr, annot=True)
plt.show()
