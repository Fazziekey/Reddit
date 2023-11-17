from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
import matplotlib.pyplot as plt

# 创建Spark会话
spark = SparkSession.builder.appName("SubredditAnalysis").getOrCreate()

# 读取JSON文件
df = spark.read.json("result.jsonl")

# 对结果进行分组和聚合
agg_df = df.groupBy("subreddit").agg(
    count(when(col("results") == "positive", True)).alias("positive_count"),
    count(when(col("results") == "neutral", True)).alias("neutral_count"),
    count(when(col("results") == "negative", True)).alias("negative_count"),
    count("*").alias("total")
)

# 计算比例
result_df = agg_df.withColumn("positive_ratio", col("positive_count") / col("total"))
result_df = result_df.withColumn("neutral_ratio", col("neutral_count") / col("total"))
result_df = result_df.withColumn("negative_ratio", col("negative_count") / col("total"))

# 收集数据用于绘图
plot_data = result_df.select("subreddit", "positive_ratio", "neutral_ratio", "negative_ratio").collect()

morandi_colors = ["#E5F6DA", "#FBEADA", "#FADCDB"]

# 绘制每个subreddit的饼图
for row in plot_data:
    labels = 'Positive', 'Neutral', 'Negative'
    sizes = [row['positive_ratio'], row['neutral_ratio'], row['negative_ratio']]
    plt.figure(figsize=(8, 6))
    # plt.pie(sizes, labels=labels, colors=morandi_colors, autopct='%1.1f%%', startangle=140)
    plt.pie(sizes, colors=morandi_colors, startangle=140, explode=(0.1, 0, 0), autopct='%1.1f%%')
    plt.legend(labels, title="Results", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title(f"Results distribution in subreddit: {row['subreddit']}")
    plt.savefig(f"plot/{row['subreddit']}.png")
    plt.clf()

# 停止Spark会话
spark.stop()




