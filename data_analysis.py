# imports
import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

import openai


def data_analysis():
    """
    Function reads preprocessed data and then flattens the embeddings using t-SNE.
    Next, it draws two charts:
    * Products ratings visualized in language using t-SNE
    * Clusters identified visualized in language 2d using t-SNE
    """

    # check the system and add files to path
    input_preprocessed_file_name = "food_reviews_embeddings_20.csv"
    # output_file_name = "food_reviews_embeddings_10.csv"
    if os.name == "posix":
        datapath = "data/"
        reportpath = "reports/"
        print("Linux")
    elif os.name == "nt":
        datapath = "data\\"
        reportpath = "reports\\"
        print("Windows")
    else:
        datapath = "data\\"
        reportpath = "reports\\"
        print("other")
    input_datapath = datapath + input_preprocessed_file_name
    # output_datapath = datapath + output_file_name


    # load data
    df = pd.read_csv(input_datapath, sep=';',index_col=0, encoding= 'ansi')

    # convert string to numpy array
    # print("length before: " + str(len(df["embedding"].iloc[0]))) # 34422
    df["embedding"] = df.embedding.apply(eval).apply(np.array)  
    # print("length after: " + str(len(df["embedding"].iloc[0]))) # 1536
    matrix = np.vstack(df.embedding.values)
    # print("size of matrix: " + str(matrix.shape)) # (10, 1536)


    # find the clusters using K-means
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels

    get_report_about_found_clusters(df, reportpath + "report.txt", n_clusters)

    mean_score_of_every_cluster = df.groupby("Cluster").Score.mean().sort_values()

    # transform data into 2 dimensions
    # tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    # prepare data for the chart
    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]

    # prepare canvas for the chart
    plt.rcParams['figure.figsize'] = [9, 8]
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Scatter plots for analysis of products ratings and comments')
    ax1.set_title("Products ratings visualized in language using t-SNE")
    ax2.set_title("Clusters identified visualized in language 2d using t-SNE")

    # draw top chart - Scores
    colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
    color_indices = df.Score.values - 1
    colormap = matplotlib.colors.ListedColormap(colors)
    ax1.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)

    # calculate the average position for each Score and mark it on the graph
    for score in [0,1,2,3,4]:
        avg_x = np.array(x)[df.Score-1==score].mean()
        avg_y = np.array(y)[df.Score-1==score].mean()
        color = colors[score]
        ax1.scatter(avg_x, avg_y, marker='x', color=color, s=100, label=str(score+1))

    # reverse order of legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(reversed(handles), reversed(labels), title='Score', loc='upper left')
    # ax1.legend()
    
    # draw bottom chart - Clusters
    for category, color in enumerate(["purple", "green", "red", "blue"]):
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        ax2.scatter(xs, ys, color=color, alpha=0.3)

        avg_x = xs.mean()
        avg_y = ys.mean()

        ax2.scatter(avg_x, avg_y, marker="x", color=color, s=100)  
        ax2.annotate(str(category), (avg_x, avg_y), textcoords="offset points", xytext=(0,5), ha='center')

    # save and show window with charts
    plt.savefig(reportpath + "report_chart.png") 
    plt.show()

    # print(df)
    # print(df.head(2))


def get_report_about_found_clusters(df, name_of_report, n_clusters):
    """
    Preparing summary about each cluster group

    Function attributes
    -------------------
    df panda : dictionary with data
    name_of_report : str
        name of file with report, optionally you can add the path to the file
    n_clusters : int
        number of found clasters

    Raises
    ------
    Exception
        If no OpenAI API kei is set as environment variable in the system.
    """

    rev_per_cluster = 3#5
    used_tokens = 0

    # open file to save report
    file_with_report = open(name_of_report, "w")

    # set your API key
    openai.api_key = os.getenv("OPENAI_API_KEY") # get API key from environment variable
    if not openai.api_key: raise Exception("There is no API kei!")

    for i in range(n_clusters):
        print(f"Cluster {i} Theme:", end=" ")
        file_with_report.write(f"Cluster {i} Theme:")

        # reviews = "\n".join(
        #     df[df.Cluster == i]
        #     .combined.str.replace("Title: ", "")
        #     .str.replace("\n\nContent: ", ":  ")
        #     .sample(rev_per_cluster, random_state=42)
        #     .values
        # )
        reviews = "\n".join(
            df[df.Cluster == i]
            .Text.sample(rev_per_cluster, random_state=42)
            .values
        )
        
        response = openai.Completion.create(
            engine="text-davinci-003",
            # prompt=f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:',
            prompt=f'Co następujące opinie klientów mają wspólnego?\n\nOpinie klientów:\n"""\n{reviews}\n"""\n\nTemat:',
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        used_tokens += response["usage"]["total_tokens"]
        print(response["choices"][0]["text"].replace("\n", ""))
        file_with_report.write(response["choices"][0]["text"].replace("\n", "") + "\n")

        sample_cluster_rows = df[df.Cluster == i].sample(rev_per_cluster, random_state=42)
        for j in range(rev_per_cluster):
            print(sample_cluster_rows.Score.values[j], end=" - ")
            file_with_report.write(str(sample_cluster_rows.Score.values[j]) + " - ")
            print(sample_cluster_rows.Text.str[:70].values[j])
            file_with_report.write(sample_cluster_rows.Text.str[:70].values[j] + "\n")

        print("-" * 100)
        file_with_report.write("-" * 100 + "\n")

    print("USED TOKENS: " + str(used_tokens))
    # close file with report
    file_with_report.close()

if __name__ == "__main__":
    data_analysis()