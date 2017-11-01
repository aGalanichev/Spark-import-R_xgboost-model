library(xgboost)
library(Matrix)
library(data.table)
library(bit64)
library(RODBC)

setwd("~/tbank")
vars <- readRDS("vars_to_keep_default.RDS")
vars <- c(vars, "cpe_type")
d <- readRDS('datadic.RDS')

ch <- odbcConnect("teradata")
DT<-as.data.table(sqlQuery(ch,"SELECT z.*, x.* FROM(
                           SELECT APPLICATION_ID, APPLICATION_DT, BKI_CREDHIST, PHONE, SPD FROM
                           (SELECT APPLICATION_ID, APPLICATION_DT, BKI_CREDHIST, PHONE, SPD FROM UAT_DM.as_sample_tb1
                           UNION ALL
                           SELECT APPLICATION_ID, APPLICATION_DT, BKI_CREDHIST, PHONE, SPD FROM UAT_DM.as_sample_tb2
                           UNION ALL
                           SELECT  APPLICATION_ID, APPLICATION_DT, 
                           CASE WHEN BKI_CREDHIST='1' THEN 'Yes' ELSE 'No' END BKI_CREDHIST, PHONE, '' SPD FROM UAT_DM.as_sample_tb_verify
                           ) a GROUP BY APPLICATION_ID, APPLICATION_DT, BKI_CREDHIST, PHONE, SPD) z
                           JOIN UAT_DM.as_dmsc x ON z.PHONE=x.MSISDN AND CAST(z.APPLICATION_DT AS DATE FORMAT 'DD.MM.YYYY')=x.report_date AND x.depth=30", dec=".", buffsize=100000,rows_at_time=1024, stringsAsFactors=T))
					   
output_vector <- DT$SPD

for (k in names(DT))
{
  if(substr(k,nchar(k)-1,nchar(k))=='.1')
    (
      k1<-substr(k,1,nchar(k)-2)
    )
  else
    (
      k1=k
    )
  if(!is.null(d[[k1]]))
  {
    print (k)
    DT[,k]<-addNA(factor(as.character(DT[,k, with=F]),levels=unlist(d[k1])),ifany=F)
  }
}

DT[,`:=`(cv1=sd_day_mou/avg_day_mou, cvr=sd_day_voice_cnt/avg_day_voice_cnt, lbc=log(bc_lifetime+1), SPD=NULL)]

drop_these_columns <- which(!(names(DT) %in% vars))
DT[, (drop_these_columns):=NULL]

DT[bc_sd_day_voice_cnt< -90 | bc_sd_day_voice_cnt>90, bc_sd_day_voice_cnt:=0]

options(na.action = na.pass)
msisdn <- DT[,1]
DT <- DT[!is.na(output_vector),-1]
output_vector <- output_vector[!is.na(output_vector)]

fmap = r2pmml::genFMap(DT)
r2pmml::writeFMap(fmap, "xgboost.fmap")

model_matrix = r2pmml::genDMatrix(output_vector, DT, "xgboost.svm")

param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",    
              "max_depth" = 4,    
              "eta" = .02,    
              "gamma" = 1.9,   
              "subsample" = .5,   
              "colsample_bytree" =.5, 
              "min_child_weight" = 6,  
              "scale_pos_weight" = 23,
              "tree_method"="hist",
              "lambda" = 3,
              "alpha" = 0
              #,"max_delta_step"= 3
)

# cv<-xgb.cv(data = model_matrix, label = output_vector, nrounds = 1000,params=param,early.stop.round=50, nfold=5)
# nrounds=which.max(cv$test.auc.mean)
nrounds <- 436

bst <- xgboost(data = model_matrix, label = output_vector, nrounds = nrounds,params=param)
res <- predict(bst, model_matrix)
res <- as.data.table(cbind(msisdn, res))
fwrite(res, 'results.csv')
#xgb.save(bst, "Default_model")

xgb.save(bst, "xgboost.model")

xgb.dump(bst, "xgboost.model.txt", fmap = "xgboost.fmap")
