import java.io.File

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.hive.HiveContext
import org.shaded.jpmml.evaluator.spark.EvaluatorUtil
import org.shaded.jpmml.evaluator.spark.TransformerBuilder

object MainObj {

  val CPE_Types = Array("C1", "C2", "C3", "C4", "C5", "C6")
  val cpe_func = udf( (cpe_type_val: String) => if(CPE_Types.contains(cpe_type_val)) cpe_type_val else null )

  def main(args: Array[String]):Unit = {

    val master = if(args.length > 0) args(0) else "local[*]"
    val pmmlPath = if(args.length > 1) args(1) else "xgb.pmml"
    val tableInput = if(args.length > 2) args(2) else "data.parquet"
    val tableOutput = if(args.length > 3) args(3) else "scoring_database.scoring_table_output"

    System.setProperty("hive.metastore.uris", "thrift://t2ru-bda-mnp-001:9083")
    System.setProperty("hive.metastore.uris", "thrift://t2ru-bda-mnp-002:9083")

    val vars = "subs_id, report_date, msisdn, lifetime, cpe_type, dev_park_size, av_dev_use, max_dev_use, avg_day_mou, " +
      "sd_day_voice_cnt, avg_ses_mou, sd_ses_mou, r5vs, cl_tail_out_msg_share, cl_smartphones_share, sms_900_cnt, " +
      "sms_900_days, sms_679_cnt, bc_sd_day_voice_cnt, bc_lifetime, sd_day_mou, avg_day_voice_cnt"

    val sparkConf = new SparkConf()
      .setMaster(master)
      .setAppName("Spark Scoring Application")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    val hiveContext = new HiveContext(sc)   //cluster


    var df = hiveContext.sql(s"select $vars from $tableInput")    // cluster
    //var df = sqlContext.read.parquet(tableInput)                    // local

    df = df.withColumn("cv1", expr("sd_day_mou/avg_day_mou"))
    df = df.withColumn("cvr", expr("sd_day_voice_cnt/avg_day_voice_cnt"))
    df = df.withColumn("lbc", expr("log(bc_lifetime+1)"))

    df = df.withColumn("model_id", expr("1"))
    df = df.withColumn("cpe_type", cpe_func(df.col("cpe_type")))

    df = df.withColumnRenamed("sms_679_cnt", "SMS_679_cnt")
    df = df.withColumnRenamed("sms_900_cnt", "SMS_900_cnt")
    df = df.withColumnRenamed("sms_900_days", "SMS_900_days")
    df = df.withColumnRenamed("r5vs", "R5VS")

    val pmmlFile = new File(pmmlPath)

    val evaluator = EvaluatorUtil.createEvaluator(pmmlFile)

    val pmmlTransformer = new TransformerBuilder(evaluator)
      .withTargetCols
      .withOutputCols
      .exploded(false)
      .build

    df.show(10)

    var pmml = pmmlTransformer
      .transform(df)
      .select("msisdn", "subs_id", "model_id", "report_date","pmml._target", "pmml.probability(1)")
      .withColumnRenamed("pmml._target", "score_type")
      .withColumnRenamed("pmml.probability(1)", "score_value")

    //pmml.write.format("csv").save(s"$tableOutput")                        // HDFS

    //pmml.show(10)                                                           // local

    pmml.registerTempTable("pmml")                                        // cluster
    hiveContext.sql(s"insert into $tableOutput select * from pmml")       // cluster
  }
}
