"""
Sentiment Analysis for Social Media

"""

install.packages(c("tidyverse", "sentimentr"))
library(tidyverse)
library(sentimentr)

# Load the dataset
data <- read_csv("tweets.csv")

# Clean the text data
data <- data %>%
  select(text, sentiment) %>%
  mutate(text = tolower(text))


# Perform sentiment analysis
data <- data %>%
  mutate(sentiment_scores = sentiment(text))
# Visualize sentiment distribution
sentiment_plot <- data %>%
  count(sentiment) %>%
  ggplot(aes(x = sentiment, y = n, fill = sentiment)) +
  geom_bar(stat = "identity") +
  labs(title = "Sentiment Distribution",
       x = "Sentiment", y = "Count") +
  theme_minimal()

print(sentiment_plot)

# Summary statistics
summary_stats <- data %>%
  group_by(sentiment) %>%
  summarise(avg_sentiment_score = mean(sentiment_scores$polarity),
            max_sentiment_score = max(sentiment_scores$polarity),
            min_sentiment_score = min(sentiment_scores$polarity))

print(summary_stats)

# Sentiment by category
sentiment_by_category <- data %>%
  group_by(category) %>%
  summarise(avg_sentiment_score = mean(sentiment_scores$polarity))

print(sentiment_by_category)

