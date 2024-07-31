import pandas as pd
import matplotlib.pyplot as plt
# s

# Read the dataset
df = pd.read_csv('C:\Pro\ANN\phong_gripper.csv', header= 0, index_col=0)  # type: ignore

# Display the original column names
print("Original column names:", df.columns.tolist())

# Strip any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Display the cleaned column names
print("Cleaned column names:", df.columns.tolist())

# Assuming the dataset has columns named 'x1', 'x2', 'x3', 'x4' for input variables
# and 'y' for the output variable. Adjust these names if necessary.
input_vars = ['x1 (mm)', 'x2 (mm)', 'x3 (mm)', 'x4 (mm)']
output_var = 'stroke of hand  (mm)'  # Replace 'y' with the actual name of your output variable

x_var = 'x1 (mm)'  # Replace with the exact name of your x3 column
y_var = 'x3 (mm)'  # Replace with the exact name of your output variable

# Calculate the correlation matrix
correlation_matrix = df[input_vars + [output_var]].corr()

# Extract the correlation values between input variables and the output variable
correlations = correlation_matrix[output_var][input_vars]

# Print the correlations
print("Correlation between input variables and output variable:")
print(correlations)

# Plotting the correlations
# plt.figure(figsize=(10, 6))
# correlations.plot(kind='bar', color='skyblue')
# plt.title('Correlation between Input Variables and Output Variable')
# plt.xlabel('Input Variables')
# plt.ylabel('Correlation Coefficient')
# plt.ylim(-1, 1)  # Correlation coefficients range from -1 to 1
# plt.axhline(0, color='black', linewidth=0.5)  # Add a horizontal line at y=0
# plt.grid(axis='y', linestyle='--', linewidth=0.7)
# plt.show()

# # Plotting the scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(df[x_var], df[y_var], color='skyblue', edgecolor='k', alpha=0.7)
# plt.title(f'Scatter Plot of {x_var} vs {y_var}')
# plt.xlabel(x_var)
# plt.ylabel(y_var)
# plt.grid(True)
# plt.show()

# fig = px.scatter(x=df['x1 (mm)'], y=df['x3 (mm)'] )
# fig.update_layout(
#     title="Correlation-based dimensionality reduction",
#     xaxis_title="x1 (mm)",
#     yaxis_title="x2 (mm)",
# )    
# fig.show()
