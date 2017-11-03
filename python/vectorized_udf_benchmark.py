from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count

from pyspark.sql.functions import pandas_udf

df = spark.range(0, 10 * 1000 * 1000).withColumn('id', (col('id') / 1000).cast('integer')).withColumn('v', rand())

df.cache()
df.count()


@udf("double")
def plus_one(v):
    return v + 1

df.withColumn('v', plus_one(df.v)).agg(count(col('v'))).show()


@pandas_udf("double")
def vectorized_plus_one(v):
    return v + 1

df.withColumn('v', vectorized_plus_one(df.v)).agg(count(col('v'))).show()


import pandas as pd
from scipy import stats

@udf('double')
def cdf(v):
    return float(stats.norm.cdf(v))

df.withColumn('cumulative_probability', cdf(df.v)).agg(count(col('cumulative_probability'))).show()


import pandas as pd
from scipy import stats

@pandas_udf('double')
def vectorized_cdf(v):
    return pd.Series(stats.norm.cdf(v))

df.withColumn('cumulative_probability', vectorized_cdf(df.v)).agg(count(col('cumulative_probability'))).show()


from pyspark.sql import Row
@udf(ArrayType(df.schema))
def substract_mean(rows):
    vs = pd.Series([r.v for r in rows])
    vs = vs - vs.mean()
    return [Row(id=rows[i]['id'], v=float(vs[i])) for i in range(len(rows))]
              
df.groupby('id').agg(collect_list(struct(df['id'], df['v'])).alias('rows')).withColumn('new_rows', substract_mean(col('rows'))).withColumn('new_row', explode(col('new_rows'))).withColumn('id', col('new_row.id')).withColumn('v', col('new_row.v')).agg(count(col('v'))).show()


@pandas_udf(df.schema)
# Input/output are both a pandas.DataFrame
def vectorized_subtract_mean(pdf):
    return pdf.assign(v=pdf.v - pdf.v.mean())

df.groupby('id').apply(vectorized_subtract_mean).agg(count(col('v'))).show()

