from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.functions import when
from pyspark.sql.functions import avg, count, round
import os
import pandas as pd

spark = SparkSession.builder \
    .appName("pyspark_smartphone_addiction_pipeline") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.sql.adaptive.enabled", "false") \
    .getOrCreate()
#Bronze Layer: Raw Data Ingestion
df = spark.read.csv("Smartphone_Usage_And_Addiction_Data.csv", header=True, inferSchema=True)
# Silver Layer: Data Cleaning and Transformation
df = df.select("user_id", "age", "gender", "daily_screen_time_hours", "sleep_hours", "academic_work_impact", "addiction_level")
df = df.limit(1500)
df = df.dropDuplicates()
df = df.dropna(how='all')
df = df.fillna('Not at all',subset='addiction_level')
df = df.withColumn("daily_screen_time_hours", df["daily_screen_time_hours"].cast(FloatType()))
df = df.withColumn("high_usage_flag",when(df["daily_screen_time_hours"] > 4, 1).otherwise(0))
df.printSchema()
# Gold Layer: Data Analysis - Aggregation
# Create output directories if they don't exist
os.makedirs("result_dataset", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Avg screen time per age group
age_group_screen_time = df.groupBy("age").agg(round(avg("daily_screen_time_hours"), 2).alias("avg_screen_time"))
age_group_screen_time.toPandas().to_csv("result_dataset/age_group_screen_time.csv", index=False)
print("Saved age_group_screen_time.csv")

# Addiction score per gender
addiction_score = df.groupBy("gender").agg(round(avg("high_usage_flag"), 2).alias("addiction_score"))
addiction_score.toPandas().to_csv("result_dataset/addiction_score.csv", index=False)
print("Saved addiction_score.csv")

# Impact of smartphone usage on academic work
academic_impact = df.groupBy("academic_work_impact").agg(count("user_id").alias("user_count"))
academic_impact.toPandas().to_csv("result_dataset/academic_impact.csv", index=False)
print("Saved academic_impact.csv")

# Save cleaned data as CSV
df.toPandas().to_csv("output/cleaned_data.csv", index=False)
print("Saved cleaned_data.csv")
print("All operations completed successfully!")
