# from openai import OpenAI

# client = OpenAI()

# try:
#     completion = client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair.",
#             },
#             {
#                 "role": "user",
#                 "content": "Compose a poem that explains the concept of recursion in programming.",
#             },
#         ],
#     )
# except Exception as e:
#     print(e)

# import matplotlib.pyplot as plt
# import numpy as np

# # Points
# x = np.linspace(-1.5, 1.5, 400)
# y = np.linspace(-1.5, 1.5, 400)
# X, Y = np.meshgrid(x, y)

# # L1 norm (Diamond)
# Z1 = np.abs(X) + np.abs(Y)

# # L2 norm (Circle)
# Z2 = np.sqrt(X**2 + Y**2)

# # Plot
# plt.figure(figsize=(10, 5))

# # L1 Norm Plot
# plt.subplot(1, 2, 1)
# plt.contour(X, Y, Z1, levels=[1], colors="blue")
# plt.title("L1 Norm (Diamond)")
# plt.grid(True)

# # L2 Norm Plot
# plt.subplot(1, 2, 2)
# plt.contour(X, Y, Z2, levels=[1], colors="red")
# plt.title("L2 Norm (Circle)")
# plt.grid(True)

# plt.show()
