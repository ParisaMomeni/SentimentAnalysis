import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a new figure with subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 30))  # 3 rows, 1 column

# Load and display the interest visualization
interest_img = mpimg.imread('output/interests/sentiment_by_interest.png')
axs[0].imshow(interest_img)
axs[0].axis('off')  # Hide axes
#axs[0].set_title('Mean Sentiment by Interest Category', fontsize=16)

# Load and display the gender visualization
gender_img = mpimg.imread('output/gender/gender_mean_sentiment_barplot.png')
axs[1].imshow(gender_img)
axs[1].axis('off')  # Hide axes
#axs[1].set_title('Gender Mean Distribution', fontsize=16)

# Load and display the account type visualization
account_img = mpimg.imread('output/account_type/account_type_sentiment_plot.png')  # Adjust the path as needed
axs[2].imshow(account_img)
axs[2].axis('off')  # Hide axes
#axs[2].set_title('Account Type Mean Distribution', fontsize=16)

# Adjust layout and save the combined figure
plt.tight_layout()
plt.savefig('output/combined_visualization.png')