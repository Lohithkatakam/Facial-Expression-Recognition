import pandas as pd
import matplotlib.pyplot as plt

# Read the emotion log file
df = pd.read_csv('emotion_log.csv')

# Count the occurrences of each emotion
summary = df['emotion'].value_counts()

# Print the emotion summary
print("Emotion Summary:")
print(summary)

# Plot the emotion frequencies in a bar chart
summary.plot(kind='bar', color='skyblue', title='Detected Emotions Frequency')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('emotion_summary.png')  # Save the plot as a PNG file
plt.show()  # Display the plot

