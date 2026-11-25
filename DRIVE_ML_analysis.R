#----------------------------------------------------------------
## DRIVE study: Leveraging machine learning to identify predictors of digital health technology use among rheumatic patients using German survey data
#
# Author: Pascal Petit
# Date of creation: 2026-11-21
# Last update: 2026-11-25
#
# Purpose
# - predict current use of digital health technologies (DHTs) among patients with rheumatic diseases (binary outcome: 1=user, 0=non-user) from 80 survey items with extreme gradient boosting (XGBoost)
# - quantify and describe feature contributions with SHAP values (overall, and within observed users / non-users)
# - run a primary analysis and two sensitivity analyses that exclude conceptually outcome-proximal predictors (sensitivity_1: domain D; sensitivity_2: domains B and D)
# - repeat the complete pipeline with ten random seeds and combine the results: pooled out-of-sample performance, across-seed SHAP, final model
#
# Inputs (folder `data_dir`, the survey data are not part of this repository)
# - DRIVE_Data_engl.csv: anonymized survey data (one row per participant; ';'-separated)
# - nomenclaturef.csv: codebook (columns Parameter, Def, categ [A-D domain])
#
# Outputs (folder `out_dir`, "Drive - ML analysis")
# - <analysis>/seed_<seed>/: per-seed results (model, hyperparameters, performance tables with 95% CIs, pre-processing logs, SHAP tables and figures, session info)
# - combined_across_seeds/ : results across seeds and analyses
#    ¤ AUROC, hyperparameter, performance (per seed, median [range], pooled out-of-sample) and SHAP tables (per seed and global)
#    ¤ Figures: ROC curves, combined SHAP figure, SHAP comparison across analyses
#    ¤ Final XGBoost model (features selected in >= 50% of seeds)
#
# Notes
# - results depend on the XGBoost/missRanger versions and may differ slightly across operating systems and thread counts
# - sessionInfo() is saved with every run
#----------------------------------------------------------------

#----------------------------------------------------------------
#### Configurations ####

#- - - -
## Disabling memory torture (a debugging option that makes R extremely slow)

gctorture(FALSE)

#- - - -
## Resetting R session

rm(list=ls())

#- - - -
## Packages needed

pack_needed<-c("data.table", "tidyverse", "broom", "splitTools", "conflicted", "caret", "MLmetrics", "xgboost", "pROC", "here", "scales", "missRanger", "patchwork", "ragg")

#- - - -
## Installing missing packages, then loading all packages needed for the analysis

is_installed<-pack_needed %in% rownames(installed.packages(all.available=TRUE))
if(any(!is_installed)){
 install.packages(pack_needed[!is_installed], repos="https://cloud.r-project.org")
}
invisible(lapply(pack_needed, library, character.only=TRUE))

#- - - -
## Resolving function-name conflicts between packages explicitly

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("slice", "dplyr")
conflict_prefer("alpha", "scales")

#- - - -
## Setting paths and working directory

data_dir<-here::here("data") # folder with the raw input files
out_dir<-here::here("Drive - ML analysis") # folder for all results
dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)

#- - - -
## Number of CPU threads used by XGBoost (capped at 4 for modest hardware)

if(isTRUE(as.logical(Sys.getenv("_R_CHECK_LIMIT_CORES_")))){
 ncores<-2L
}else{
 ncores<-max(1L, min(4L, parallel::detectCores()))
}

#- - - -
## Default random seed (the pipeline function accepts any other seed)

seed<-123L

#----------------------------------------------------------------
#### Creating helper functions ####

#- - - -
## Negation of %in% ("not in")

`%ni%`<-Negate("%in%")

#- - - -
## Min-max scaling of a vector into [0, 1]

std1<-function(x){
 rng<-max(x, na.rm=TRUE) - min(x, na.rm=TRUE)
 if(is.na(rng) || rng==0) return(rep(0, length(x)))
 (x - min(x, na.rm=TRUE)) / rng
}

#- - - -
## Formatting a number for display

test_format1<-function(x, digits=2){
  x<-as.numeric(x)
  if(length(x) != 1 || is.na(x) || is.infinite(x)) return("")
  if(x==0) return("0")
  a<-abs(x)
  if(a < 0.01 || a >= 10000) return(format(signif(x, digits), scientific=TRUE))
  if(a >= 1000) return(as.character(round(x)))
  as.character(signif(x, digits))
}

# Vectorised version (works inside mutate() without rowwise())
test_format<-function(x, digits=2) vapply(as.numeric(x), test_format1, character(1), digits=digits)

# Formats every non-integer numeric column of a table (3 significant digits by default)
fmt_tbl<-function(df, digits=3){
  df %>% mutate(across(where(~ is.numeric(.x) && !all(.x == round(.x), na.rm=TRUE)),
                       ~ test_format(.x, digits)))
}

# Drop-in replacement for write.table(): formats at write time only
write_tbl<-function(x, file, ...){
  utils::write.table(fmt_tbl(as.data.frame(x)), file, sep=";", col.names=TRUE, row.names=FALSE)
}

#- - - -
## Custom ggplot2 theme used for all figures

theme_Gaia<-function(){
 theme_bw() +
  theme(strip.text=element_text(size=12, colour="black", face="bold"),
   strip.background=element_rect(fill="#CAE1FF", colour="black"),
   axis.text=element_text(size=12, color="black"),
   axis.title=element_text(size=12, face="bold", color="black"),
   legend.text=element_text(size=12),
   legend.title=element_text(size=12, face="bold"),
   axis.line=element_line(color="black", linewidth=0.1))
}

#- - - -
## Variance inflation factors computed from the predictor correlation matrix

compute_vif<-function(X){
 R<-stats::cor(X)
 R[is.na(R)]<-0 # to handle constant columns
 diag(R)<-1
 inv<-tryCatch(solve(R), error=function(e) solve(R + diag(1e-6, ncol(R)))) # to handle perfect collinearity
 vif<-diag(inv)
 vif[vif < 0]<-Inf # numerical safeguard for near-singular R
 stats::setNames(vif, colnames(X))
}

#- - - -
## Class-imbalance handling
#
# - "none" (default, recommended) leaves the training data and the loss function untouched, which keeps predicted probabilities calibrated
# - "balanced" sets scale_pos_weight=n(0) / n(1)

get_spw<-function(y, class_weighting="none"){
 if(class_weighting=="balanced") sum(y==0) / sum(y==1) else 1
}

#- - - -
## Mean absolute SHAP value per feature (global importance) of an XGBoost model
#
# - evaluated on the rows of X
# - the model's bias/intercept term is excluded

mean_abs_shap<-function(model, X){
 contrib<-predict(model, X, predcontrib=TRUE)
 contrib<-contrib[, colnames(contrib) %ni% c("BIAS", "Intercept", "(Intercept)"),
  drop=FALSE]
 colMeans(abs(contrib))
}

#- - - -
## Random-search hyperparameter tuning with stratified k-fold CV
#
# - candidate configurations are drawn without replacement from a grid
# - the number of boosting rounds is tuned by early stopping on the AUROC
# - returns the best configuration and the full table of tried configurations

tune_xgb<-function(X, y, seed, n_configs=30L, k=5L, max_rounds=300L, nthread=1L, class_weighting="none"){
 spw <-get_spw(y, class_weighting)
 grid<-expand.grid(subsample=c(0.5, 0.7, 0.9),
  colsample_bytree=c(0.5, 0.7, 0.9),
  min_child_weight=c(1, 5, 10),
  eta=c(0.05, 0.1, 0.3), # learning rate
  max_depth=c(3, 5, 7))
 
 # Random subset of the grid (limited for computational efficiency)
 set.seed(seed)
 grid<-grid[sample.int(nrow(grid), min(n_configs, nrow(grid))), , drop=FALSE]
 
 # Stratified folds, with invert=TRUE that returns the indices of the held-out part of each fold
 cv_folds<-splitTools::create_folds(factor(y), k=k, type="stratified",
  seed=seed, invert=TRUE)
 dtrain<-xgboost::xgb.DMatrix(data=X, label=y)
 
 res<-vector("list", nrow(grid))
 for(i in seq_len(nrow(grid))){
  params<-c(as.list(grid[i, ]),
 list(objective="binary:logistic", eval_metric="auc", scale_pos_weight=spw, nthread=nthread, seed=seed))
  cv<-xgboost::xgb.cv(params=params, data=dtrain, nrounds=max_rounds, folds=cv_folds, early_stopping_rounds=10,
  verbose=0)
  log_i<-as.data.frame(cv$evaluation_log)
  best_iter<-which.max(log_i$test_auc_mean)
  res[[i]]<-data.frame(grid[i, ], nrounds=log_i$iter[best_iter], cv_auc=log_i$test_auc_mean[best_iter])
 }
 res<-do.call(rbind, res)
 list(best=res[which.max(res$cv_auc), ], all=res)
}

#- - - -
## Fitting one XGBoost model with a given configuration

fit_xgb<-function(X, y, best, seed, nthread=1L, class_weighting="none"){
 params<-list(objective="binary:logistic", eval_metric="auc",
   subsample=best$subsample, colsample_bytree=best$colsample_bytree,
   min_child_weight=best$min_child_weight, eta=best$eta,
   max_depth=best$max_depth,
   scale_pos_weight=get_spw(y, class_weighting),
   nthread=nthread, seed=seed)
 xgboost::xgb.train(params=params, data=xgboost::xgb.DMatrix(data=X, label=y),
  nrounds=best$nrounds, verbose=0)
}

#- - - -
## Bootstrap distribution of the mean of x (resampling observations with replacement)

boot_means<-function(x, B=1000L){
 n<-length(x)
 if(n < 2) return(rep(NA_real_, B))
 replicate(B, mean(x[sample.int(n, n, replace=TRUE)]))
}

#- - - -
## Summary of |SHAP| values per feature
#
# - mean_val=mean |SHAP| (the global importance of the feature)
# - CI_low/CI_high=95% percentile-bootstrap confidence interval of that mean (resampling test participants; quantifies the sampling uncertainty of the mean, conditional on the fitted model)
# - P2.5/P97.5=empirical 2.5th/97.5th percentiles of the individual |SHAP| values

summarize_abs_shap<-function(grouped_df, B=1000L){
 grouped_df %>%
  dplyr::summarize(mean_val=mean(abs(shap_value), na.rm=TRUE),
  P2.5=stats::quantile(abs(shap_value), 0.025, na.rm=TRUE),
  P97.5=stats::quantile(abs(shap_value), 0.975, na.rm=TRUE),
  boot=list(boot_means(abs(shap_value), B)),
  .groups ="drop") %>%
  dplyr::mutate(CI_low =purrr::map_dbl(boot, ~ stats::quantile(.x, 0.025, na.rm=TRUE, names=FALSE)),
   CI_high=purrr::map_dbl(boot, ~ stats::quantile(.x, 0.975, na.rm=TRUE, names=FALSE))) %>%
  dplyr::select(-boot)
}

#- - - -
## Calibration measures of predicted probabilities p for 0/1 outcomes y
#
# - Brier score
# - calibration intercept (calibration-in-the-large, ideal=0)
# - calibration slope (ideal=1)

calibration_metrics<-function(p, y){
 p_clip<-pmin(pmax(p, 1e-6), 1 - 1e-6)
 lp<-stats::qlogis(p_clip)
 slope<-tryCatch(unname(stats::coef(stats::glm(y ~ lp, family=stats::binomial))[2]),
  error=function(e) NA_real_)
 intcpt<-tryCatch(unname(stats::coef(stats::glm(y ~ offset(lp), family=stats::binomial))[1]),
  error=function(e) NA_real_)
 c(Brier=mean((p - y)^2), Calibration_intercept=intcpt, Calibration_slope=slope)
}

#- - - -
## Learn the preprocessing on X (numeric matrix, raw NAs allowed)
#
# - all-NA columns -> missRanger -> near-zero variance -> VIF
# - do_nzv=FALSE and/or vif_threshold=Inf switch a step off

preprocess_fit<-function(X, vif_threshold=2.5, seed=123L, do_nzv=TRUE){
 all_na<-colnames(X)[colSums(!is.na(X))==0]
 X<-X[, colnames(X) %ni% all_na, drop=FALSE]
 n_missing<-sum(is.na(X))
 imp_fit<-NULL
 if(n_missing > 0){
  imp_fit<-missRanger::missRanger(as.data.frame(X), pmm.k=3, num.trees=100, maxiter=5,
   seed=seed, verbose=0, keep_forests=TRUE)
  X<-as.matrix(imp_fit$data)
 }
 cols_imp<-colnames(X)  # columns the imputation model knows
 medians<-apply(X, 2, stats::median, na.rm=TRUE)
 
 nzv<-character(0)
 if(do_nzv){
  is_bin<-apply(X, 2, function(x) length(unique(x)) <= 2)
  if(any(is_bin)){
 Xb<-X[, is_bin, drop=FALSE]
 nzv<-colnames(Xb)[caret::nearZeroVar(Xb)]
  }
  if(length(nzv) > 0) X<-X[, colnames(X) %ni% nzv, drop=FALSE]
 }
 
 vif_log<-data.frame(removed=character(0), vif=numeric(0))
 if(is.finite(vif_threshold)){
  repeat{
 v<-compute_vif(X)
 if(max(v) <= vif_threshold || ncol(X) <= 2) break
 worst<-names(which.max(v))
 vif_log<-rbind(vif_log, data.frame(removed=worst, vif=unname(max(v))))
 X<-X[, colnames(X) != worst, drop=FALSE]
  }
 }
 list(X=X, features=colnames(X), cols_imp=cols_imp, all_na=all_na, imp_fit=imp_fit,
  medians=medians, nzv=nzv, vif_log=vif_log, n_missing=n_missing)
}

#- - - -
## Apply a learned preprocessing to new data (never re-learns anything)

preprocess_apply<-function(prep, Xnew, seed=123L){
 Xnew<-Xnew[, prep$cols_imp, drop=FALSE]
 if(!is.null(prep$imp_fit) && anyNA(Xnew)){
  Xnew<-as.matrix(predict(prep$imp_fit, newdata=as.data.frame(Xnew), seed=seed, verbose=0))
 }
 n_fallback<-sum(is.na(Xnew)) # safety net: training median
 if(n_fallback > 0){
  for(j in seq_len(ncol(Xnew))){
 na_idx<-is.na(Xnew[, j])
 if(any(na_idx)) Xnew[na_idx, j]<-prep$medians[colnames(Xnew)[j]]
  }
 }
 Xnew<-Xnew[, prep$features, drop=FALSE]
 attr(Xnew, "n_fallback")<-n_fallback
 Xnew
}

#- - - -
## Saving a figure as PDF + PNG with the same size

save_fig<-function(plot, path_stub, width, height, dpi=300){
 ggplot2::ggsave(paste0(path_stub, ".pdf"), plot, width=width, height=height, units="cm", limitsize=FALSE)
 ggplot2::ggsave(paste0(path_stub, ".png"), plot, width=width, height=height, units="cm", dpi=dpi, limitsize=FALSE, device=ragg::agg_png, bg="white")
}

#- - - -
## Confusion-matrix metrics with 95% CIs
#
# - Accuracy, Sensitivity, Specificity, PPV, NPV: exact Clopper-Pearson CI
# - Balanced accuracy, F1 (and Brier, PR-AUC if prob is given): percentile bootstrap over participants
# - PPV/NPV CIs are conditional on the prevalence of the test sample

cm_metrics_ci<-function(y, pred, prob=NULL, B=1000L, seed=1L){
  cnt<-function(y, p) c(tp=sum(p==1 & y==1), fp=sum(p==1 & y==0),
                        fn=sum(p==0 & y==1), tn=sum(p==0 & y==0))
  k<-cnt(y, pred)
  n<-length(y)
  ex<-function(x, m){
    x<-unname(x); m<-unname(m)
    if(m==0) return(c(NA_real_, NA_real_, NA_real_))
    c(x/m, unname(stats::binom.test(x, m)$conf.int))
  }
  # Proportions: exact Clopper-Pearson CI
  exact<-rbind(Accuracy=ex(k["tp"] + k["tn"], n),
               Sensitivity=ex(k["tp"], k["tp"] + k["fn"]),
               Specificity=ex(k["tn"], k["tn"] + k["fp"]),
               PPV=ex(k["tp"], k["tp"] + k["fp"]),
               NPV=ex(k["tn"], k["tn"] + k["fn"]),
               Prevalence=ex(k["tp"] + k["fn"], n),
               Detection_rate=ex(k["tp"], n),
               Detection_prevalence=ex(k["tp"] + k["fp"], n),
               No_information_rate=ex(max(k["tp"] + k["fn"], k["fp"] + k["tn"]), n))
  
  # Other metrics: percentile bootstrap over participants
  stat<-function(y, p, pr){
    k<-cnt(y, p); n<-sum(k)
    sens<-k["tp"] / (k["tp"] + k["fn"]); spec<-k["tn"] / (k["tn"] + k["fp"])
    po<-(k["tp"] + k["tn"]) / n
    pe<-((k["tp"] + k["fp"]) * (k["tp"] + k["fn"]) + (k["fn"] + k["tn"]) * (k["fp"] + k["tn"])) / n^2
    out<-c(Balanced_accuracy=unname((sens + spec) / 2),
           F1=unname(2 * k["tp"] / (2 * k["tp"] + k["fp"] + k["fn"])),
           Kappa=unname((po - pe) / (1 - pe)))
    if(!is.null(pr)){
      cal<-suppressWarnings(calibration_metrics(pr, y))   # Brier, calibration intercept, calibration slope
      out<-c(out, PRAUC=MLmetrics::PRAUC(y_pred=pr, y_true=y),
             GainAUC=MLmetrics::GainAUC(y_pred=pr, y_true=y), cal)
    }
    out
  }
  est<-stat(y, pred, prob)
  set.seed(seed)
  bs<-replicate(B, {
    i<-sample.int(n, n, replace=TRUE)
    stat(y[i], pred[i], if(is.null(prob)) NULL else prob[i])
  })
  boot<-cbind(est,
              apply(bs, 1, stats::quantile, 0.025, na.rm=TRUE),
              apply(bs, 1, stats::quantile, 0.975, na.rm=TRUE))
  res<-rbind(exact, boot)
  tibble::tibble(metric=rownames(res), estimate=unname(res[, 1]),
                 CI_low=unname(res[, 2]), CI_high=unname(res[, 3]),
                 CI_method=c(rep("Clopper-Pearson exact", nrow(exact)),
                             rep("percentile bootstrap", nrow(boot))))
}

#- - - -
## Two-stage bootstrap of the across-seed mean of mean|SHAP|
#
# - vals = list (named by seed) of the |SHAP| vectors of one feature
# - a seed absent from the list = feature not in that model (contribution 0)
# - stage 1: resample seeds
# - stage 2: resample participants within each drawn seed

boot_two_stage<-function(vals, all_seeds, B){
 nm<-as.character(all_seeds); S<-length(nm)
 draws<-replicate(B, {
  ss<-sample(nm, S, replace=TRUE)
  mean(vapply(ss, function(s){
 v<-vals[[s]]
 if(is.null(v) || length(v) < 1) return(0)
 mean(v[sample.int(length(v), length(v), replace=TRUE)])
  }, numeric(1)))
 })
 stats::quantile(draws, c(0.025, 0.975), names=FALSE)
}

#- - - -
## Replace raw feature names by readable labels

relabel<-function(x, labels){
 if(is.null(labels)) return(x)
 lab<-labels %>% select(Parameter, Def) %>% distinct(Parameter, .keep_all=TRUE)
 x$feature<-dplyr::coalesce(lab$Def[match(x$feature, lab$Parameter)], x$feature)
 x
}

#- - - -
## Global SHAP summary across seeds for one analysis
#
# - point estimate = mean over seeds of the per-seed mean|SHAP| (0 if the feature is not in that seed's final model)
# - 95% CI = two-stage bootstrap (seeds, then participants)

shap_across_seeds<-function(results, analysis, labels=NULL, n_boot=500L, seed=1L){
 res<-results[vapply(results, function(r) r$analysis_name==analysis, logical(1))]
 all_seeds<-vapply(res, function(r) as.numeric(r$seed), numeric(1))
 long<-dplyr::bind_rows(lapply(res, function(r) dplyr::mutate(r$shap_full, seed=r$seed)))
 long_grp<-dplyr::bind_rows(
  dplyr::mutate(long, outcome="overall"),
  dplyr::mutate(long, outcome=dplyr::if_else(obs_outcome==1, "DHT user", "non DHT user")))
 
 set.seed(seed)
 agg<-long_grp %>%
  group_by(outcome, feature) %>%
  summarize(vals=list(split(abs(shap_value), seed)), .groups="drop") %>%
  mutate(per_seed=purrr::map(vals, function(v){
 m<-stats::setNames(rep(0, length(all_seeds)), as.character(all_seeds))
 mm<-vapply(v, mean, numeric(1)); m[names(mm)]<-mm; m}),
 mean_val=purrr::map_dbl(per_seed, mean),
 SD_across_seeds=purrr::map_dbl(per_seed, stats::sd),
 n_seeds_in_model=purrr::map_int(vals, length),
 ci=purrr::map(vals, ~ boot_two_stage(.x, all_seeds, n_boot)),
 CI_low=purrr::map_dbl(ci, 1), CI_high=purrr::map_dbl(ci, 2),
 mean_SHAP=paste0(test_format(mean_val), " (", test_format(CI_low), "; ", test_format(CI_high), ")")) %>%
  group_by(outcome) %>%
  mutate(SHAP_per=mean_val / sum(mean_val) * 100) %>% ungroup() %>%
  select(-vals, -per_seed, -ci) %>%
  arrange(outcome, desc(mean_val)) %>%
  mutate(analysis=analysis)
 
 list(summary=relabel(agg, labels), long=relabel(long, labels))
}

#- - - -
## Pooled out-of-sample predictions across seeds for one analysis
#
# - each participant's probability = mean over the seeds where he/she was in the test set (never used to train that model)
# - threshold = median of the seed-specific thresholds (all chosen on training data)

pooled_oob<-function(results, analysis, B=1000L, seed=1L){
 res<-results[vapply(results, function(r) r$analysis_name==analysis, logical(1))]
 oob<-dplyr::bind_rows(lapply(res, function(r) r$test_pred)) %>%
  group_by(orig_id) %>%
  summarize(y=dplyr::first(y), prob=mean(prob), n_models=dplyr::n(), .groups="drop")
 thr<-stats::median(vapply(res, function(r) r$threshold, numeric(1)))
 roc_o<-pROC::roc(oob$y, oob$prob, levels=c(0, 1), direction="<", quiet=TRUE)
 auc<-pmin(as.numeric(pROC::ci.auc(roc_o, method="delong")), 1)
 pred<-as.integer(oob$prob > thr)
 list(oob=oob, threshold=thr, auc=auc,
      counts=tibble::tibble(TP=sum(pred==1 & oob$y==1), FP=sum(pred==1 & oob$y==0),
                            FN=sum(pred==0 & oob$y==1), TN=sum(pred==0 & oob$y==0)),
      metrics=cm_metrics_ci(oob$y, pred, oob$prob, B=B, seed=seed))
}

#- - - -
## Combined figure for SHAP
#
# - beeswarm (left)
# - mean|SHAP| barplots for users / non-users (right)

plot_shap_combined<-function(bees, bars, top_n=20L, w_ratio=c(3, 2), n_seeds=NULL, wrap=40){
  bees<-dplyr::mutate(bees, feature=as.character(feature))
  bars<-dplyr::mutate(bars, feature=as.character(feature))
  top<-bars %>% filter(outcome=="overall") %>% arrange(desc(mean_val)) %>% slice_head(n=top_n)
  ord<-top$feature
  
  # Axis label = feature (+ number of seeds in which it was selected)
  new_lab<-if(!is.null(n_seeds) && "n_seeds_in_model" %in% names(top)){
    paste0(top$feature, " [", top$n_seeds_in_model, "/", n_seeds, "]")
  }else top$feature
  lab<-stats::setNames(new_lab, ord)
  lev<-rev(unname(lab)) # coord_flip: first level at the bottom
  to_f<-function(x) factor(unname(lab[x]), levels=lev)
  
  bees<-bees %>% filter(feature %in% ord) %>% group_by(feature) %>%
    mutate(fv=std1(feature_value)) %>% ungroup() %>%
    mutate(feature=to_f(feature), panel="Overall")
  lab_bees<-top %>% mutate(feature=to_f(feature), panel="Overall")
  y_lab<-max(bees$shap_value, na.rm=TRUE) * 1.05
  
  bars<-bars %>% filter(outcome != "overall", feature %in% ord) %>%
    tidyr::complete(outcome, feature=ord, fill=list(mean_val=0, SHAP_per=0)) %>%
    mutate(feature=to_f(feature))
  
  p1<-ggplot(bees, aes(x=feature, y=shap_value, colour=fv)) +
    geom_hline(yintercept=0, colour="darkgrey") +
    geom_jitter(width=0.15, height=0, size=0.7, alpha=0.6) +
    geom_text(data=lab_bees, aes(x=feature, y=y_lab, label=mean_SHAP),
              hjust=0, size=3.5, colour="black", inherit.aes=FALSE) +   # mean |SHAP| (95% CI)
    scale_colour_gradient(name="Feature value", low="blue", high="red", limits=c(0, 1),
                          breaks=c(0, 1), labels=c("Low", "High"), na.value="lightgrey") +
    scale_x_discrete(labels=function(x) stringr::str_wrap(x, wrap)) +
    scale_y_continuous(expand=expansion(mult=c(0.02, 0.5))) +
    coord_flip() + facet_wrap(~ panel) +
    labs(x="Feature", y="SHAP value (impact on model output)") +
    theme_Gaia() + theme(legend.position="right", legend.key.height=unit(2.5, "cm"))
  
  p2<-ggplot(bars, aes(x=feature, y=SHAP_per)) +
    geom_col(fill="grey", colour="black", width=0.7) +
    geom_text(aes(label=sprintf("%.1f", SHAP_per)), hjust=-0.15, size=3.5) +
    scale_y_continuous(expand=expansion(mult=c(0, 0.2))) +
    coord_flip() + facet_wrap(~ outcome) +
    labs(x=NULL, y="Relative contribution to model output (%)") +
    theme_Gaia() + theme(axis.text.y=element_blank(), axis.ticks.y=element_blank())
  
  p1 + p2 + patchwork::plot_layout(widths=w_ratio)
}

#- - - -
## Tile plot comparing the global relative contribution across analyses

plot_shap_tile<-function(global_overall, n_top=20L, main_name="main_analysis", n_seeds=10L, wrap=45){
  lab_an<-c(main_analysis="Main analysis",
            sensitivity_1="Sensitivity 1\n(without D)",
            sensitivity_2="Sensitivity 2\n(without B, D)")
  keep<-global_overall %>% group_by(analysis) %>% arrange(desc(mean_val)) %>%
    slice_head(n=n_top) %>% ungroup() %>% pull(feature) %>% unique()
  ord<-global_overall %>% filter(feature %in% keep) %>%
    group_by(feature) %>%
    summarize(o=max(SHAP_per[analysis==main_name], 0, na.rm=TRUE), m=max(SHAP_per), .groups="drop") %>%
    arrange(o, m) %>% pull(feature)
  df<-global_overall %>% filter(feature %in% keep) %>%
    tidyr::complete(analysis, feature) %>% # NA = feature never selected
    mutate(analysis=factor(unname(lab_an[as.character(analysis)]), levels=unname(lab_an)),
           feature=factor(feature, levels=ord),
           txt=ifelse(is.na(SHAP_per), "", sprintf("%.1f\n(%d/%d)", SHAP_per, n_seeds_in_model, n_seeds)))
  ggplot(df, aes(x=analysis, y=feature, fill=SHAP_per)) +
    geom_tile(colour="black") +
    geom_text(aes(label=txt), size=3, lineheight=0.9) +
    scale_y_discrete(expand=c(0,0),labels=function(x) stringr::str_wrap(x, wrap)) +
    scale_x_discrete(expand=c(0,0),position="top") +
    scale_fill_gradient(low="white", high="#40B696", na.value="grey90", name="Relative\ncontribution (%)") +
    labs(x=NULL, y=NULL) +
    theme_Gaia() + 
    theme(panel.grid=element_blank(), axis.text.y=element_text(size=10))
}

#- - - -
## Main pipeline function
#
#' run the complete XGBoost + SHAP pipeline.
#'
#' @param data Data frame with a column named "outcome" (0/1) and all candidate predictors as numeric/binary columns (missing values allowed and imputed internally)
#' @param analysis_name Label of the analysis (e.g., "main_analysis", "sensitivity_1")
#' @param seed Random seed (data split, CV folds, imputation, random search, XGBoost, bootstrap)
#' @param n_top Number of predictors kept in the final model
#' @param n_configs Number of random hyperparameter configurations per tuning run
#' @param vif_threshold Predictors are removed iteratively until all VIFs <= this value
#' @param class_weighting "none" (default: no reweighting or resampling) or "balanced" (scale_pos_weight=n0/n1)
#' @param threshold_method "youden" (default: threshold maximizing sensitivity + specificity - 1 in out-of-fold training predictions) or "fixed" (uses fixed_threshold)
#' @param fixed_threshold Threshold used when threshold_method="fixed".
#' @param n_boot Number of bootstrap resamples for the SHAP confidence intervals
#' @param labels Optional codebook with columns "Parameter" (feature name) and "Def" (readable label) used in the figures
#' @param n_threads Number of CPU threads for XGBoost
#' @param out_root Root folder for results
#' @return A list with the key results

run_xgb_pipeline<-function(data,
   analysis_name,
   seed=123L,
   n_top=30L,
   n_configs=30L,
   vif_threshold=2.5,
   class_weighting=c("none", "balanced"),
   threshold_method=c("youden", "fixed"),
   fixed_threshold=0.5,
   n_boot=1000L,
   labels=NULL,
   n_threads=ncores,
   out_root=out_dir){
 
 class_weighting <-match.arg(class_weighting)
 threshold_method<-match.arg(threshold_method)
 
 #--------------------------------------------------
 ## Setting up folders and file labels
 
 set.seed(seed)
 out_path<-file.path(out_root, analysis_name, paste0("seed_", seed))
 dir.create(out_path, showWarnings=FALSE, recursive=TRUE)
 tag<-paste0(analysis_name, "_seed", seed) # appended to every file name
 save_to<-function(file) file.path(out_path, file)
 message("[", tag, "] Starting pipeline")
 
 # Saving the software versions for reproducibility
 writeLines(capture.output(sessionInfo()), save_to(paste0("sessionInfo_", tag, ".txt")))
 
 #--------------------------------------------------
 ## Input checks and conversion to data.table
 
 dataset<-data.table::as.data.table(data)
 stopifnot("The data must contain a column named 'outcome'"="outcome" %in% names(dataset))
 dataset<-dataset[!is.na(outcome)] # outcome must be observed
 stopifnot("The outcome must be coded 0/1"=all(dataset$outcome %in% c(0, 1)))
 
 target_col <-"outcome"
 feature_cols<-setdiff(names(dataset), target_col)
 
 #--------------------------------------------------
 ## Stratified 70/30 train-test split (stratified by the outcome)
 
 data_split<-splitTools::partition(y=factor(dataset[[target_col]]), p=c(train=0.7, test=0.3),
   type="stratified", seed=seed)
 
 X_train<-as.matrix(dataset[data_split$train, ..feature_cols])
 y_train<-as.integer(dataset[[target_col]][data_split$train])
 X_test<-as.matrix(dataset[data_split$test, ..feature_cols])
 y_test<-as.integer(dataset[[target_col]][data_split$test])
 stopifnot(is.numeric(X_train)) # all predictors must be numeric
 n_candidate_initial<-ncol(X_train)
 
 #--------------------------------------------------
 ## Pre-processing learned on the training set only (full training set -> used for the test set and the final model)
 
 X_train_raw<-X_train # kept: the nested CV below re-learns the preprocessing inside every fold
 X_test_raw <-X_test
 
 prep_full<-preprocess_fit(X_train_raw, vif_threshold=vif_threshold, seed=seed)
 X_train<-prep_full$X
 X_test <-preprocess_apply(prep_full, X_test_raw, seed=seed)
 n_fallback_test<-attr(X_test, "n_fallback")
 
 candidate_features<-prep_full$features
 n_candidate_final<-length(candidate_features)
 nzv_names<-prep_full$nzv
 vif_log<-prep_full$vif_log
 imputation_log<-data.frame(missing_values_train=prep_full$n_missing,
  missing_values_test=sum(is.na(X_test_raw[, prep_full$cols_imp])),
  test_values_filled_by_training_median=n_fallback_test)
 
 message("[", tag, "] Candidate predictors: ", n_candidate_initial, " -> ",
   n_candidate_final, " (all-NA removed: ", length(prep_full$all_na),
   "; near-zero variance removed: ", length(nzv_names),
   "; VIF removed: ", nrow(vif_log), ")")
 
 #--------------------------------------------------
 ## Nested cross-validation on the training set
 # 
 # - outer loop (10 folds): honest performance estimate of the whole pipeline (tuning -> SHAP ranking -> top-feature model), using only outer-training data
 # - inner loop (5 folds): random-search hyperparameter tuning
 
 # invert=TRUE returns the indices of the held-out (validation) part
 outer_val_folds<-splitTools::create_folds(factor(y_train), k=10, type="stratified", seed=seed, invert=TRUE)
 outer_results<-vector("list", length(outer_val_folds))
 outer_importance<-vector("list", length(outer_val_folds))
 
 for(o in seq_along(outer_val_folds)){
  val_idx<-outer_val_folds[[o]]
  tr_idx<-setdiff(seq_len(nrow(X_train_raw)), val_idx)
  y_tr<-y_train[tr_idx]
  y_val<-y_train[val_idx]
  
  # Preprocessing re-learned on the outer-training fold only, then applied to the validation fold
  prep_o<-preprocess_fit(X_train_raw[tr_idx, , drop=FALSE], vif_threshold=vif_threshold, seed=seed + o)
  X_tr <-prep_o$X
  X_val<-preprocess_apply(prep_o, X_train_raw[val_idx, , drop=FALSE], seed=seed + o)
  
  # Step 1: tune and fit a model with all candidate predictors
  tune_all<-tune_xgb(X_tr, y_tr, seed=seed + o, n_configs=n_configs, nthread=n_threads, class_weighting=class_weighting)
  m_all<-fit_xgb(X_tr, y_tr, tune_all$best, seed=seed + o, nthread=n_threads, class_weighting=class_weighting)
  auc_all<-as.numeric(pROC::auc(pROC::roc(y_val, predict(m_all, X_val), levels=c(0, 1), direction="<", quiet=TRUE)))
  
  # Step 2: rank predictors by mean |SHAP| in the outer-training data only
  imp_all<-mean_abs_shap(m_all, X_tr)
  imp_pos<-sort(imp_all[imp_all > 0], decreasing=TRUE)
  top_feats<-names(imp_pos)[seq_len(min(n_top, length(imp_pos)))]
  outer_importance[[o]]<-data.frame(fold=o, feature=names(imp_all), mean_abs_shap=as.numeric(imp_all), in_top=names(imp_all) %in% top_feats)
  
  # Step 3: re-tune and refit using only the top predictors, then evaluate on the outer validation fold (never seen in steps 1-3)
  tune_top<-tune_xgb(X_tr[, top_feats, drop=FALSE], y_tr, seed=seed + o, n_configs=n_configs, nthread=n_threads,
 class_weighting=class_weighting)
  m_top<-fit_xgb(X_tr[, top_feats, drop=FALSE], y_tr, tune_top$best, seed=seed + o, nthread=n_threads, class_weighting=class_weighting)
  p_val<-predict(m_top, X_val[, top_feats, drop=FALSE])
  auc_top<-as.numeric(pROC::auc(pROC::roc(y_val, p_val, levels=c(0, 1), direction="<", quiet=TRUE)))
  
  outer_results[[o]]<-data.frame(Fold=o,
   n_candidates=ncol(X_tr),
   AUROC_all_features=signif(auc_all, 3),
   AUROC_top_features=signif(auc_top, 3),
   Brier_top_features=signif(mean((p_val - y_val)^2), 3),
   top_var=paste(top_feats, collapse=";"),
   top_val=paste(signif(imp_pos[top_feats], 4), collapse=";"))
  message("[", tag, "] Outer fold ", o, "/", length(outer_val_folds), " done (AUROC top features=", signif(auc_top, 3), ")")
 }
 outer_summary  <-dplyr::bind_rows(outer_results)
 outer_importance<-dplyr::bind_rows(outer_importance)
 
 #--------------------------------------------------
 ## Final feature selection and final model (training set only)
 
 n_outer<-length(outer_val_folds)
 
 importance_summary<-outer_importance %>%
  group_by(feature) %>%
  summarize(mean_abs_shap=sum(mean_abs_shap) / n_outer,  # absent fold = 0
   n_folds_available=dplyr::n(), n_folds_in_top=sum(in_top), .groups="drop") %>%
  arrange(desc(mean_abs_shap))
 
 final_features<-importance_summary %>%
  filter(feature %in% candidate_features, mean_abs_shap > 0) %>%  # must exist in the full-training preprocessing
  slice_head(n=n_top) %>%
  pull(feature)
 
 X_train_final<-X_train[, final_features, drop=FALSE]
 X_test_final<-X_test[, final_features, drop=FALSE]
 
 # Re-tuning on the full training set with the selected predictors
 tune_final<-tune_xgb(X_train_final, y_train, seed=seed, n_configs=n_configs,
 nthread=n_threads, class_weighting=class_weighting)
 final_params<-tune_final$best
 xgb_final<-fit_xgb(X_train_final, y_train, final_params, seed=seed, nthread=n_threads, class_weighting=class_weighting)
 
 xgboost::xgb.save(xgb_final, save_to(paste0("XGBoost_model_", tag, ".ubj")))
 
 # Complete record of the hyperparameters of the final model, including the fixed settings
 final_params_full<-tibble::tibble(analysis=analysis_name, seed=seed,
   subsample=final_params$subsample, colsample_bytree=final_params$colsample_bytree,
   min_child_weight=final_params$min_child_weight, eta=final_params$eta,
   max_depth=final_params$max_depth, nrounds=final_params$nrounds,
   scale_pos_weight=get_spw(y_train, class_weighting),
   class_weighting=class_weighting,
   objective="binary:logistic", eval_metric="auc (tuning)",
   inner_cv_auc_of_selected_configuration=signif(final_params$cv_auc, 3))
 
 #--------------------------------------------------
 ## Classification threshold (chosen on training data only)
 #
 # - out-of-fold predicted probabilities of the final configuration are obtained by 10-fold CV within the training set
 # - the threshold maximizing Youden's J (sensitivity + specificity - 1) is prevalence-independent and treats false positives and false negatives symmetrically
 # - the test set is never used to choose the threshold
 
 oof_folds<-splitTools::create_folds(factor(y_train), k=10, type="stratified", seed=seed + 999, invert=TRUE)
 oof_prob<-rep(NA_real_, length(y_train))
 for(f in oof_folds){
  tr<-setdiff(seq_along(y_train), f)
  prep_f<-preprocess_fit(X_train_raw[tr, , drop=FALSE], vif_threshold=Inf, seed=seed, do_nzv=FALSE)
  Xtr_f<-prep_f$X[, final_features, drop=FALSE]
  Xva_f<-preprocess_apply(prep_f, X_train_raw[f, , drop=FALSE], seed=seed)[, final_features, drop=FALSE]
  m<-fit_xgb(Xtr_f, y_train[tr], final_params, seed=seed, nthread=n_threads, class_weighting=class_weighting)
  oof_prob[f]<-predict(m, Xva_f)
 }
 roc_oof<-pROC::roc(y_train, oof_prob, levels=c(0, 1), direction="<", quiet=TRUE)
 oof_auroc<-as.numeric(pROC::auc(roc_oof))
 
 if(threshold_method=="youden"){
  threshold<-as.numeric(unlist(pROC::coords(roc_oof, x="best", best.method="youden", ret="threshold")))[1]
  if(!is.finite(threshold)) threshold<-0.5
 }else{
  threshold<-fixed_threshold
 }
 message("[", tag, "] Classification threshold (", threshold_method, "): ", signif(threshold, 3))
 
 #--------------------------------------------------
 ## One-time evaluation on the held-out test set
 
 test_prob<-predict(xgb_final, X_test_final)
 test_class<-as.integer(test_prob > threshold)
 
 # AUROC with a DeLong 95% CI (capped at 1)
 roc_obj<-pROC::roc(y_test, test_prob, levels=c(0, 1), direction="<", quiet=TRUE)
 auc_ci<-as.numeric(pROC::ci.auc(roc_obj, method="delong")) # lower, AUC, upper
 auc_ci<-pmin(auc_ci, 1)
 AUC_CI<-paste0(signif(auc_ci[2], 3), " (", signif(auc_ci[1], 3), "; ", signif(auc_ci[3], 3), ")")
 
 # Confusion matrix and all derived metrics
 cm<-caret::confusionMatrix(factor(test_class, levels=c(0, 1)), factor(y_test, levels=c(0, 1)),
  mode="everything", positive="1")
 
 # Accuracy with its exact binomial 95% CI
 acc_ci<-paste0(signif(cm$overall["Accuracy"], 3), " (",
   signif(cm$overall["AccuracyLower"], 3), "; ",
   signif(cm$overall["AccuracyUpper"], 3), ")")
 
 # Calibration of the predicted probabilities (only meaningful without class weighting)
 cal<-calibration_metrics(test_prob, y_test)
 
 # 95% CIs of the confusion-matrix metrics + raw counts
 cm_ci<-cm_metrics_ci(y_test, test_class, test_prob, B=n_boot, seed=seed)
 cm_counts<-tibble::tibble(TP=sum(test_class==1 & y_test==1),
   FP=sum(test_class==1 & y_test==0),
   FN=sum(test_class==0 & y_test==1),
   TN=sum(test_class==0 & y_test==0))
 
 # Class distribution of the full analysis sample
 n_cases<-sum(dataset$outcome==1)
 n_noncases<-sum(dataset$outcome==0)
 n_cases_txt<-paste0(n_cases, " (", signif(n_cases/(n_cases + n_noncases) * 100, 3), "%)")
 n_noncases_txt<-paste0(n_noncases, " (", signif(n_noncases/(n_cases + n_noncases) * 100, 3), "%)")
 
 # One-row summary table of the test-set performance
 
 byclass_tbl<-tibble::as_tibble(as.list(cm$byClass), .name_repair="minimal")
 names(byclass_tbl)<-gsub(" ", "_", names(byclass_tbl))
 overall_tbl<-tibble::as_tibble(as.list(cm$overall)) %>% select(-Accuracy)
 
 confus_mat_temp<-dplyr::bind_cols(
  tibble::tibble(analysis=analysis_name, seed=seed,
 n_train=length(y_train), n_test=length(y_test),
 n_candidate_predictors=n_candidate_final,
 n_final_predictors=length(final_features),
 outcome1_n_pct_full_sample=n_cases_txt,
 outcome0_n_pct_full_sample=n_noncases_txt,
 AUC=AUC_CI, AUC_value=auc_ci[2],
 training_out_of_fold_AUC=signif(oof_auroc, 3),
 threshold_method=threshold_method,
 threshold=signif(threshold, 3),
 Accuracy=acc_ci,
 PRAUC =MLmetrics::PRAUC(y_pred=test_prob, y_true=y_test),
 GainAUC=MLmetrics::GainAUC(y_pred=test_prob, y_true=y_test),
 Brier=unname(cal["Brier"]),
 Calibration_intercept=unname(cal["Calibration_intercept"]),
 Calibration_slope=unname(cal["Calibration_slope"])),
  byclass_tbl, overall_tbl)
 
 confus_mat_caret<-broom::tidy(cm) %>% select(-any_of("class"))
 
 #--------------------------------------------------
 ## Saving tables
 
  write_tbl(outer_summary, save_to(paste0("CV summary table_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(confus_mat_temp, save_to(paste0("Confusion matrix_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(confus_mat_caret, save_to(paste0("caret confusion matrix_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(final_params_full, save_to(paste0("Best hyperparameters_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(tune_final$all, save_to(paste0("Hyperparameter search_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(importance_summary, save_to(paste0("Feature selection across outer folds_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(vif_log, save_to(paste0("VIF removal log_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(data.frame(near_zero_variance_removed=nzv_names), save_to(paste0("Near-zero variance removed_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
  write_tbl(imputation_log, save_to(paste0("Imputation log_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE)
 
 #--------------------------------------------------
 ## SHAP analysis (final model, held-out test set)
 
 # SHAP values of every test participant (rows) and feature (columns)
 contr<-predict(xgb_final, X_test_final, predcontrib=TRUE)
 shap_dt<-data.table::as.data.table(contr)
 
 # Dropping the bias (intercept) term
 shap_dt<-shap_dt[, names(shap_dt) %ni% c("BIAS", "Intercept", "(Intercept)"), with=FALSE]
 shap<-tibble::as_tibble(shap_dt)
 
 # Features ordered by mean |SHAP| (used for the saved SHAP score table)
 mean_shap_score<-colMeans(abs(shap_dt))
 mean_shap_score<-mean_shap_score[order(mean_shap_score, decreasing=TRUE)]
 shap_score_sub <-shap_dt[, names(mean_shap_score), with=FALSE]
 
 # Long format: one row per participant x feature
 shap_long<-shap %>%
  mutate(id=row_number(), obs_outcome=y_test) %>%
  pivot_longer(cols=-c(id, obs_outcome), names_to="feature", values_to="shap_value")
 
 # Summary of |SHAP| overall and separately within observed users / non-users
 #
 # - the stratified summaries describe contributions within each observed group
 # - they are not separate models or within-group predictions
 
 set.seed(seed)  # reproducible bootstrap
 Shap_val_overall<-shap_long %>% group_by(feature) %>% summarize_abs_shap(B=n_boot) %>% mutate(outcome="overall")
 Shap_val_group<-shap_long %>% group_by(obs_outcome, feature) %>% summarize_abs_shap(B=n_boot) %>%
  mutate(outcome=if_else(obs_outcome==1, "DHT user", "non DHT user")) %>%
  select(-obs_outcome)
 
 # mean_SHAP="mean |SHAP| (95% bootstrap CI)", used in tables and figures
 Shap_val<-bind_rows(Shap_val_overall, Shap_val_group) %>%
  arrange(outcome, desc(mean_val)) %>%
  rowwise() %>%
  mutate(mean_SHAP=paste0(test_format(mean_val), " (", test_format(CI_low), "; ", test_format(CI_high), ")"),
  mean_SHAP=if_else(mean_SHAP %in% c("0 (0; 0)", "0e+00 (0e+00; 0e+00)"), "0", mean_SHAP)) %>% ungroup()
 
 # Per-participant table: feature value + SHAP value + group summaries
 feature_values<-X_test_final %>% tibble::as_tibble() %>% rowid_to_column("id") %>%
  pivot_longer(-id, names_to="feature", values_to="feature_value")
 shap_by_id<-shap_long %>% select(id, feature, shap_value)
 
 # Everything needed to aggregate across seeds (raw feature names, original row index of each participant)
 shap_full<-feature_values %>%
  left_join(shap_long, by=c("id", "feature")) %>%
  mutate(orig_id=data_split$test[id])
 test_pred<-data.frame(orig_id=data_split$test, prob=test_prob, y=y_test)
 
 plot_data<-feature_values %>%
  left_join(shap_by_id, by=c("id", "feature")) %>%
  left_join(Shap_val, by="feature") # adds overall + group summaries
 
 # Replacing raw feature names by readable labels from the codebook (if given)
 if(!is.null(labels)){
  label_data<-labels %>% select(Parameter, Def) %>% rename(feature=Parameter)
  plot_data<-plot_data %>%
 left_join(label_data, by="feature") %>%
 mutate(feature=dplyr::coalesce(Def, feature)) %>%
 select(-Def)
 }
 
 # Summary table (one row per outcome group x feature), used for the bar plot
 shap_summary_tbl<-plot_data %>%
  select(outcome, feature, mean_val, CI_low, CI_high, P2.5, P97.5, mean_SHAP) %>% distinct() %>%
  filter(!is.na(mean_val), mean_val != 0) %>%
  group_by(outcome) %>% arrange(desc(mean_val)) %>% ungroup()
 
 #--------------------------------------------------
 ## Figure 1: SHAP dispersion (beeswarm-type) plot, all participants
 
 plot_overall<-plot_data %>% filter(outcome=="overall") %>%
  group_by(feature) %>%
  mutate(feature_value2=std1(feature_value)) %>%  # per-feature 0-1 scaling for coloring
  ungroup()
 
 bornes_shap<-plot_overall %>% summarize(min=min(shap_value, na.rm=TRUE), max=max(shap_value, na.rm=TRUE))
 
 plot_overall<-plot_overall %>% filter(mean_SHAP != "0")
 feature_order<-plot_overall %>% select(feature, mean_val) %>% distinct() %>% arrange(mean_val)
 plot_overall$feature<-factor(plot_overall$feature, levels=feature_order$feature)
 label_tbl<-plot_overall %>% select(feature, mean_SHAP) %>% distinct()
 
 n_feat<-nlevels(plot_overall$feature)
 fig_h<-max(14, 0.8 * n_feat + 6)  # cm, grows with the number of features
 
 plot1<-ggplot2::ggplot(plot_overall, ggplot2::aes(x=feature, y=shap_value, colour=feature_value2)) +
  ggplot2::geom_jitter(width=0.1, height=0, size=1) +
  scale_color_gradient(name="Feature value", low="blue", high="red",
  limits=c(0, 1), breaks=c(0, 1), labels=c("Low", "High"),
  na.value="lightgrey") +
  ggplot2::geom_hline(yintercept=0, colour="darkgrey") +
  scale_x_discrete("Feature") +
  scale_y_continuous("SHAP value (impact on model output)",
 expand=expansion(mult=c(0.02, 0.45))) + # free space on the right for the labels
  ggplot2::coord_flip() +
  geom_text(data=label_tbl, aes(x=feature, y=bornes_shap$max * 1.05, label=mean_SHAP),
   size=3.5, hjust=0, col="black", inherit.aes=FALSE) +
  theme_Gaia() +
  theme(legend.key.height=unit(5, "cm"))
 
 save_fig(plot1, save_to(paste0("SHAP_dispersion_", tag)), width=30, height=60)
 
 #--------------------------------------------------
 ## Figure 2: relative contribution of each feature (% of total mean |SHAP|)
 
 shap_summary_tbl<-shap_summary_tbl %>% group_by(outcome) %>%
  mutate(SHAP_per=abs(mean_val) / sum(abs(mean_val)) * 100) %>% ungroup()
 
 order_overall<-shap_summary_tbl %>% filter(outcome=="overall") %>% arrange(desc(SHAP_per))
 shap_summary_tbl$feature<-factor(shap_summary_tbl$feature,
   levels=rev(unique(order_overall$feature)))
 
 plot2<-ggplot(shap_summary_tbl, aes(y=feature, x=SHAP_per)) +
  geom_bar(stat="identity", col="black", fill="grey") +
  geom_text(aes(label=sprintf("%.1f", SHAP_per)), hjust=-0.15, size=3.5) +
  scale_x_continuous("Relative contribution to model output (%)",
 expand=expansion(mult=c(0, 0.15))) +
  scale_y_discrete("Feature") +
  facet_wrap(. ~ outcome) +
  theme_Gaia() + theme(legend.position="bottom")
 
 save_fig(plot2, save_to(paste0("SHAP_contribution_", tag)), width=40, height=60)
 
 #--------------------------------------------------
 ## Figure 3: beeswarm + barplots for users / non-users
 
 plot3<-plot_shap_combined(bees=plot_overall, bars=shap_summary_tbl, top_n=20)
 save_fig(plot3, save_to(paste0("SHAP_combined_", tag)), width=50, height=max(14, 0.8 * min(20, n_feat) + 6))
 
 #--------------------------------------------------
 ## Saving SHAP results
 
 write_tbl(Shap_val, save_to(paste0("Shap_val_", tag, ".csv"))) # |SHAP| summary per feature and group
 write_tbl(shap_summary_tbl, save_to(paste0("test_tmp_", tag, ".csv"))) # table behind Figure 2
 data.table::fwrite(plot_overall, save_to(paste0("test_", tag, ".txt")), sep=";") # data behind Figure 1
 data.table::fwrite(shap, save_to(paste0("shap_", tag, ".txt")), sep=";") # raw SHAP values (test set)
 data.table::fwrite(shap_score_sub, save_to(paste0("shap_score_sub_", tag, ".txt")), sep=";") # raw SHAP values, ordered by importance
  write_tbl(cm_ci, save_to(paste0("Confusion matrix metrics with 95CI_", tag, ".csv")), sep=";", col.names=TRUE, row.names=FALSE) # confusion matrix metrics with 95CI
 
 message("[", tag, "] Finished. Test-set AUROC=", AUC_CI)
 
 #--------------------------------------------------
 ## Returning the key objects
 
 invisible(list(analysis_name=analysis_name,
   seed=seed,
   auroc=auc_ci[2],
   auroc_ci=auc_ci[c(1, 3)],
   test_metrics=confus_mat_temp,
   caret_confusion=cm,
   threshold=threshold,
   outer_cv_summary=outer_summary,
   final_features=final_features,
   feature_selection=importance_summary,
   final_params=final_params_full,
   vif_removed=vif_log,
   near_zero_variance=nzv_names,
   imputation_log=imputation_log,
   shap_summary=Shap_val,
   model=xgb_final,
   split=data_split,
   test_metrics_ci=cm_ci,
   cm_counts=cm_counts,
   test_pred=test_pred,
   shap_full=shap_full,
   output_path=out_path))
}

#- - - -
## Combining everything into one final model

fit_final_model<-function(data, results, analysis, min_seed_frac=0.5, seed=123L, n_configs=30L, class_weighting="none", n_threads=ncores){
 
 res<-results[vapply(results, function(r) r$analysis_name==analysis, logical(1))]
 sel<-table(unlist(lapply(res, function(r) r$final_features)))
 feats<-names(sel)[sel >= min_seed_frac * length(res)] # features kept in >= 50% of the seeds
 
 dataset<-data.table::as.data.table(data)[!is.na(outcome)]
 y<-as.integer(dataset$outcome)
 Xraw<-as.matrix(dataset[, setdiff(names(dataset), "outcome"), with=FALSE])
 prep<-preprocess_fit(Xraw, vif_threshold=Inf, seed=seed, do_nzv=FALSE) # imputation only; features already chosen
 feats<-intersect(feats, prep$features)
 X<-prep$X[, feats, drop=FALSE]
 message("Max VIF among final features: ", signif(max(compute_vif(X)), 3))
 
 tune<-tune_xgb(X, y, seed=seed, n_configs=n_configs, nthread=n_threads, class_weighting=class_weighting)
 model<-fit_xgb(X, y, tune$best, seed=seed, nthread=n_threads, class_weighting=class_weighting)
 
 po<-pooled_oob(results, analysis)
 roc_o<-pROC::roc(po$oob$y, po$oob$prob, levels=c(0, 1), direction="<", quiet=TRUE)
 thr<-as.numeric(unlist(pROC::coords(roc_o, x="best", best.method="youden", ret="threshold")))[1]
 
 d<-file.path(out_dir, "Combined_across_seeds"); dir.create(d, showWarnings=FALSE, recursive=TRUE)
 xgboost::xgb.save(model, file.path(d, paste0("Final_XGBoost_model_", analysis, ".ubj")))
  write_tbl(data.frame(feature=feats, n_seeds_selected=as.integer(sel[feats])),
   file.path(d, paste0("Final_features_", analysis, ".csv")), sep=";", row.names=FALSE)
  write_tbl(tune$best, file.path(d, paste0("Final_hyperparameters_", analysis, ".csv")), sep=";", row.names=FALSE)
 
 list(model=model, features=feats, params=tune$best, threshold=thr,
  performance_pooled_oob=po$metrics, auc_pooled_oob=po$auc,
  shap_mean_abs=mean_abs_shap(model, X))
}

#----------------------------------------------------------------
#### Data import and pre-processing (outcome-independent, no learning) ####

#- - - -
## Importing the codebook and the survey data

codebook<-as_tibble(read.csv(file.path(data_dir, "nomenclaturef.csv"), sep=";"))
drive<-as_tibble(read.csv2(file.path(data_dir, "DRIVE_Data_engl.csv")))

#- - - -
## Removing survey time stamps

drive<-drive %>% select(-c(Date_started, Date_end))

#- - - -
## One-hot encoding of categorical predictors

var_categ<-c("Q1", "Q2", "Q3", "Disease", "Gender", "Education_level", "qualif", "City", "region")

for(i in seq_along(var_categ)){
 
 # extracting the ID and the current categorical variable
 temp<-drive %>% dplyr::select(ID, which(colnames(drive)==var_categ[i]))
 colnames(temp)[2]<-"test"
 
 # creating one 0/1 indicator column per category, e.g., "Gender_female"
 temp<-temp %>%
  mutate(tempo=paste(var_categ[i], "_", test, sep="")) %>%
  dplyr::select(-test) %>%
  mutate(region2=1) %>%
  complete(ID, tempo, fill=list(region2=0)) %>% distinct %>%
  spread(key=tempo, value=region2)
 
 # replacing the original column by its indicator columns
 drive<-drive %>% dplyr::select(-c(which(colnames(drive)==var_categ[i]))) %>% distinct
 drive<-left_join(drive, temp, by="ID")
}

# Removing one reference category per variable to prevent perfect multicollinearity among the indicator columns
drive<-drive %>% select(-c(Q1_, Q2_, Q3_, Disease_, Gender_, Education_level_NA, qualif_NA, City_NA, region_))

#- - - -
## Removing the ID column

drive2<-drive %>% select(-c(ID))

#- - - -
## Renaming the outcome (DHT use: 1=user, 0=non-user) and placing it first

clean_data<-drive2
colnames(clean_data)[which(colnames(clean_data)=="DHS_use")]<-"outcome"
clean_data<-clean_data %>% select(outcome, everything())

#----------------------------------------------------------------
#### Data analysis - Descriptive profiles of reported reasons for non-use among non-users ####

# Filtering non-users
nonuser_data<-clean_data %>% filter(outcome==0)

# Selecting relevant variables
nonuser_data<-nonuser_data %>%
              select(Q8,Q66,Q68,Q69,Q9,Q64,Q65,Q62, Q63,Q61,Q5, Q6, Q7, Q70,Q67,Age,Gender_Female,qualif_1) %>%
              mutate(ID=seq(1,nrow(nonuser_data),1)) %>%
              pivot_longer(cols=-ID)

# Creating categories
nonuser_data<-nonuser_data %>% mutate(name=case_when(name %in% c("Q8","Q66","Q68")~"access",
                                                     name %in% c("Q69","Q9", "Q64", "Q65")~"skill_usability",
                                                     name %in% c("Q62","Q63")~"info",
                                                     name %in% c("Q61")~"privacy",
                                                     name %in% c("Q5","Q6","Q7","Q70")~"relevance",
                                                     name %in% c("Q67")~"cost",
                                                     T~name))

demo_nonuser<-nonuser_data %>% filter(name %in% c("Age","Gender_Female","qualif_1")) %>% pivot_wider(names_from=name,values_from=value)
nonuser_data<-nonuser_data %>% filter(name %ni% c("Age","Gender_Female","qualif_1")) 

nonuser_data<-left_join(demo_nonuser,nonuser_data,by="ID")

# Counting number of non users per barrier catagories
nonuser_data<-nonuser_data %>% group_by(ID,name) %>% mutate(value=sum(value)) %>% mutate(value=if_else(value>0,1,0)) %>% distinct %>%
              group_by(ID) %>% mutate(n_categ=sum(value)) %>% ungroup

# Characteristics of each profile, except those with no response
nonuser_data %>% group_by(name,value) %>% summarize(n=n(),
                                              mean_age=mean(Age,na.rm=T),
                                              sd=sd(Age,na.rm=T),
                                              n_qualif=sum(qualif_1),
                                              n_female=sum(Gender_Female)) %>%
  filter(value==1) %>%
  ungroup %>%
  mutate(per=n/153*100,
         per_qualif=n_qualif/153*100,
         per_female=n_female/153*100)

# Characteristics of the no response profile
nonuser_data %>% filter(n_categ==0) %>% select(-name) %>% distinct %>% summarize(n=n(),
                                                                                 mean_age=mean(Age,na.rm=T),
                                                                                 sd=sd(Age,na.rm=T),
                                                                                 n_qualif=sum(qualif_1),
                                                                                 n_female=sum(Gender_Female)) %>%
  ungroup %>%
  mutate(per=n/153*100,
         per_qualif=n_qualif/153*100,
         per_female=n_female/153*100)
  

#----------------------------------------------------------------
#### Data analysis - ML ####

#- - - -
## Creating vectors of variables to consider for each sensitivity analysis

# Sensitivity analysis 1 removes all domain D predictors, while sensitivity analysis 2 removes all domain B and D predictors 
domain_B<-codebook %>% filter(categ=='B') %>% select(Parameter) %>% distinct %>% pull
domain_D<-codebook %>% filter(categ=='D') %>% select(Parameter) %>% distinct %>% pull


#- - - -
## Creating datasets for each analysis: primary analysis + the two sensitivity analyses

analyses<-list(main_analysis=clean_data,
   sensitivity_1=clean_data %>% select(-all_of(domain_D)),   # without domain D
   sensitivity_2=clean_data %>% select(-all_of(c(domain_B, domain_D)))) # without domains B and D

#- - - -
## Ten different seeds (each changes the split, folds, imputation, search and XGBoost seed)

seeds<-c(123, 234, 345, 456, 567, 678, 789, 890, 901, 1012)

#- - - -
## Running the analysis

results<-list()
for(a in names(analyses)){
 for(s in seeds){
  results[[paste0(a, "_seed", s)]]<-run_xgb_pipeline(data=analyses[[a]], analysis_name=a, seed=s, labels=codebook)
 }
}

#- - - -
## Output folder for combined results

dir_comb<-file.path(out_dir, "Combined_across_seeds")
dir.create(dir_comb, showWarnings=FALSE, recursive=TRUE)

#- - - -
## Summarizing test-set AUROC across seeds and the change relative to the primary model

auc_by_seed<-purrr::map_dfr(results, function(r){
 tibble::tibble(analysis=r$analysis_name, seed=r$seed, AUROC=r$auroc,
   n_final_predictors=length(r$final_features))
})

auc_by_seed<-auc_by_seed %>%
 left_join(auc_by_seed %>% filter(analysis=="main_analysis") %>%
   select(seed, AUROC_main=AUROC), by="seed") %>%
 mutate(delta_AUROC_vs_main=AUROC - AUROC_main)

auc_summary<-auc_by_seed %>%
 group_by(analysis) %>%
 summarize(median_AUROC=median(AUROC), 
           mean_AUROC=mean(AUROC),
           sd_AROC=sd(AUROC),
           p2.5_AUROC=quantile(AUROC,0.025),
           p97.5_AUROC=quantile(AUROC,0.975),,
           min_AUROC=min(AUROC), 
           max_AUROC=max(AUROC),
  median_delta_vs_main=median(delta_AUROC_vs_main),
  mean_delta_vs_main=mean(delta_AUROC_vs_main),
  .groups="drop")

auc_by_seed
auc_summary

 write_tbl(auc_by_seed, file.path(dir_comb, "AUROC_by_analysis_and_seed.csv"), sep=";", row.names=FALSE)
 write_tbl(auc_summary, file.path(dir_comb, "AUROC_summary_across_seeds.csv"), sep=";", row.names=FALSE)
 
 #- - - -
 ## Pooled delta-AUROC
 
 auc_fast<-function(y, p){# rank-based AUROC (Mann-Whitney)
   r<-rank(p)
   n1<-sum(y==1)
   n0<-sum(y==0)
   (sum(r[y==1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
 }
 
 delta_auc_pooled<-purrr::map_dfr(c("sensitivity_1", "sensitivity_2"), function(a){
   m<-pooled$main_analysis$oob
   s<-pooled[[a]]$oob
   stopifnot(identical(m$orig_id, s$orig_id), identical(m$y, s$y))
   set.seed(1)
   d<-replicate(2000, {i<-sample.int(nrow(m), replace=TRUE); auc_fast(s$y[i], s$prob[i]) - auc_fast(m$y[i], m$prob[i])})
   rm_<-pROC::roc(m$y, m$prob, levels=c(0, 1), direction="<", quiet=TRUE)
   rs_<-pROC::roc(s$y, s$prob, levels=c(0, 1), direction="<", quiet=TRUE)
   tibble::tibble(comparison=paste0(a, " vs main_analysis"),
                  delta_AUROC=auc_fast(s$y, s$prob) - auc_fast(m$y, m$prob),
                  CI_low=unname(stats::quantile(d, 0.025)), CI_high=unname(stats::quantile(d, 0.975)),
                  DeLong_paired_p=pROC::roc.test(rm_, rs_, method="delong", paired=TRUE)$p.value)
 })
 write_tbl(delta_auc_pooled, file.path(dir_comb, "Delta_AUROC_pooled_vs_main.csv"))

#- - - -
## Table 1: best hyperparameters by analysis and seed (+ summary)

 hp_cols<-c("subsample", "colsample_bytree", "min_child_weight", "eta", "max_depth", "nrounds")
 
 hp_all<-purrr::map_dfr(results, ~ .x$final_params)
 hp_by_seed<-hp_all %>% select(analysis, seed, all_of(hp_cols))
 inner_cv_by_seed<-hp_all %>% select(analysis, seed, inner_cv_auc_of_selected_configuration)  # performance, not a hyperparameter
 
 hp_summary<-hp_by_seed %>% group_by(analysis) %>%
   summarize(across(all_of(hp_cols),
                    ~ paste0(test_format(median(.x), 3), " [", test_format(min(.x), 3), "-", test_format(max(.x), 3), "]")),
             .groups="drop")
 
 write_tbl(hp_by_seed, file.path(dir_comb, "Hyperparameters_by_analysis_and_seed.csv"))
 write_tbl(hp_summary, file.path(dir_comb, "Hyperparameters_summary_across_seeds.csv"))
 write_tbl(inner_cv_by_seed, file.path(dir_comb, "Inner_CV_AUC_by_analysis_and_seed.csv"))
 
 # Final hyperparameter table
 final_models<-lapply(names(analyses), function(a) fit_final_model(analyses[[a]], results, a))
 names(final_models)<-names(analyses)
 
 final_hp_tbl<-purrr::map_dfr(names(final_models), function(a)
   final_models[[a]]$params %>% select(all_of(hp_cols)) %>% mutate(analysis=a)) %>%
   pivot_longer(-analysis, names_to="Hyperparameter", values_to="value") %>%
   mutate(value=test_format(value, 3)) %>%
   pivot_wider(names_from=analysis, values_from=value)
 
 write_tbl(final_hp_tbl, file.path(dir_comb, "Final_hyperparameters_all_analyses.csv"))

#- - - -
## Table 2: confusion-matrix metrics by analysis and seed, + global

 cm_long<-purrr::map_dfr(results, function(r)
   dplyr::mutate(r$test_metrics_ci, analysis=r$analysis_name, seed=as.character(r$seed),
                 threshold=r$threshold, auc=r$auroc, auc_low=r$auroc_ci[1], auc_high=r$auroc_ci[2]))
 
 fmt<-function(e, l, h) paste0(test_format(e, 3), " (", test_format(l, 3), "; ", test_format(h, 3), ")")
 rng<-function(x) paste0(test_format(stats::median(x), 3), " [", test_format(min(x), 3), "-", test_format(max(x), 3), "]")
 
 meta_by_seed<-cm_long %>% distinct(analysis, seed, threshold, auc, auc_low, auc_high)
 counts_by_seed<-purrr::map_dfr(results, ~ dplyr::mutate(.x$cm_counts, analysis=.x$analysis_name, seed=as.character(.x$seed)))
 
 # one row per seed: estimate (95% CI)
 cm_by_seed<-cm_long %>% mutate(txt=fmt(estimate, CI_low, CI_high)) %>%
   select(analysis, seed, metric, txt) %>%
   pivot_wider(names_from=metric, values_from=txt) %>%
   left_join(meta_by_seed %>% mutate(threshold=test_format(threshold, 3), AUC=fmt(auc, auc_low, auc_high)) %>%
               select(analysis, seed, threshold, AUC), by=c("analysis", "seed")) %>%
   left_join(counts_by_seed %>% mutate(across(c(TP, FP, FN, TN), as.character)), by=c("analysis", "seed"))
 
 # global, descriptive: median [range] of the seed-specific estimates (stability, not a CI)
 cm_range<-cm_long %>% group_by(analysis, metric) %>% summarize(txt=rng(estimate), .groups="drop") %>%
   pivot_wider(names_from=metric, values_from=txt) %>%
   left_join(meta_by_seed %>% group_by(analysis) %>% summarize(threshold=rng(threshold), AUC=rng(auc), .groups="drop"), by="analysis") %>%
   left_join(counts_by_seed %>% group_by(analysis) %>% summarize(across(c(TP, FP, FN, TN), rng), .groups="drop"), by="analysis") %>%
   mutate(seed="median [range] across seeds")
 
 # global, inferential: pooled out-of-sample predictions (1 per participant)
 pooled<-lapply(names(analyses), pooled_oob, results=results)
 names(pooled)<-names(analyses)
 cm_pooled<-purrr::map_dfr(names(pooled), function(a){
   p<-pooled[[a]]
   p$metrics %>% mutate(txt=fmt(estimate, CI_low, CI_high)) %>% select(metric, txt) %>%
     pivot_wider(names_from=metric, values_from=txt) %>%
     mutate(analysis=a, seed="pooled out-of-sample",
            threshold=test_format(p$threshold, 3), AUC=fmt(p$auc[2], p$auc[1], p$auc[3]),
            TP=as.character(p$counts$TP), FP=as.character(p$counts$FP),
            FN=as.character(p$counts$FN), TN=as.character(p$counts$TN))
 })
 
 cm_table<-bind_rows(cm_by_seed, cm_range, cm_pooled) %>% relocate(analysis, seed, threshold, AUC) %>% arrange(analysis)
 
 write_tbl(cm_table, file.path(dir_comb, "Confusion_matrix_by_analysis_and_seed.csv"))

#- - - -
## Table 3 + figures: SHAP by analysis and seed, + global

shap_global<-lapply(names(analyses), function(a) shap_across_seeds(results, a, labels=codebook, n_boot=500L))
names(shap_global)<-names(analyses)
shap_global_tbl<-bind_rows(lapply(shap_global, `[[`, "summary"))

write_tbl(shap_global_tbl, file.path(dir_comb, "SHAP_global_across_seeds.csv"), sep=";", row.names=FALSE)

 shap_by_seed<-purrr::map_dfr(results, ~ dplyr::mutate(.x$shap_summary, analysis=.x$analysis_name, seed=.x$seed)) %>% relabel(codebook)
 
 shap_wide<-shap_by_seed %>% filter(outcome=="overall") %>%
   select(analysis, feature, seed, mean_SHAP) %>%
   pivot_wider(names_from=seed, names_prefix="seed_", values_from=mean_SHAP,
               values_fill="not in final model") %>%
   left_join(shap_global_tbl %>% filter(outcome=="overall") %>% 
               select(analysis, feature, GLOBAL=mean_SHAP, n_seeds_in_model, SD_across_seeds), by=c("analysis", "feature")) %>%
   arrange(analysis, desc(n_seeds_in_model))
 
 write_tbl(shap_global_tbl, file.path(dir_comb, "SHAP_global_across_seeds.csv"))
 write_tbl(shap_by_seed, file.path(dir_comb, "SHAP_long_by_analysis_and_seed.csv"))
 write_tbl(shap_wide, file.path(dir_comb, "SHAP_wide_by_analysis_and_seed.csv"))

# Global combined figure (beeswarm + user/non-user bars) per analysis

for(a in names(analyses)){
   g<-shap_global[[a]]
   p<-plot_shap_combined(bees=g$long %>% select(feature, shap_value, feature_value), bars=g$summary, top_n=20, n_seeds=length(seeds))
   save_fig(p, file.path(dir_comb, paste0("SHAP_combined_all_seeds_", a)), width=42, height=35)
}

# Tile plot comparing the analyses (global, all seeds)

tile_df<-shap_global_tbl %>% filter(outcome=="overall")
p_tile<-plot_shap_tile(tile_df, n_top=20, n_seeds=length(seeds))
n_tile<-nlevels(p_tile$data$feature)
save_fig(p_tile, file.path(dir_comb, "SHAP_tile_comparison_analyses"), width=26, height=max(14, 0.9 * n_tile + 5))

#- - - -
## ROC figure

roc_points<-function(y, p){
  r<-pROC::roc(y, p, levels=c(0, 1), direction="<", quiet=TRUE)
  data.frame(fpr=1 - r$specificities, tpr=r$sensitivities) %>% arrange(fpr, tpr)
}

plot_roc_analysis<-function(results, analysis, pooled_a, title){
  res<-results[vapply(results, function(r) r$analysis_name==analysis, logical(1))]
  seed_curves<-purrr::map_dfr(res, ~ dplyr::mutate(roc_points(.x$test_pred$y, .x$test_pred$prob),
                                                   seed=as.character(.x$seed)))
  aucs<-vapply(res, function(r) r$auroc, numeric(1))
  overall<-roc_points(pooled_a$oob$y, pooled_a$oob$prob)
  
  ggplot() +
    geom_abline(slope=1, intercept=0, colour="grey60", linetype="dashed") +
    geom_path(data=seed_curves, aes(x=fpr, y=tpr, group=seed), colour="steelblue", alpha=0.5, linewidth=0.5) +
    geom_path(data=overall, aes(x=fpr, y=tpr), colour="black", linewidth=1.3) +
    annotate("text", x=0.98, y=0.14, hjust=1, size=4, fontface="bold",
             label=paste0("Overall (pooled out-of-sample): AUROC ", test_format(pooled_a$auc[2], 3),
                          " (", test_format(pooled_a$auc[1], 3), "; ", test_format(pooled_a$auc[3], 3), ")")) +
    annotate("text", x=0.98, y=0.05, hjust=1, size=4, colour="steelblue4",
             label=paste0("Each seed: median AUROC ", test_format(median(aucs), 3),
                          " [", test_format(min(aucs), 3), "-", test_format(max(aucs), 3), "]")) +
    coord_equal(xlim=c(0, 1), ylim=c(0, 1), expand=FALSE) +
    labs(title=title, x="1 - Specificity", y="Sensitivity") +
    theme_Gaia() + theme(plot.title=element_text(face="bold", size=12))
}

roc_titles<-c(main_analysis="Main analysis",
              sensitivity_1="Sensitivity 1 (without domain D)",
              sensitivity_2="Sensitivity 2 (without domains B and D)")

roc_plots<-lapply(names(analyses), function(a) plot_roc_analysis(results, a, pooled[[a]], roc_titles[[a]]))
names(roc_plots)<-names(analyses)

p_roc<-roc_plots$main_analysis/roc_plots$sensitivity_1/roc_plots$sensitivity_2

save_fig(p_roc, file.path(dir_comb, "ROC_curves_all_analyses"), width=50, height=50)

#- - - -
## Combining everything into one final model

final_main<-fit_final_model(clean_data, results, "main_analysis")
final_main<-fit_final_model(clean_data, results, "sensitivity_1")
final_main<-fit_final_model(clean_data, results, "sensitivity_2")

#----------------------------------------------------------------
#### Session information (reproducibility) ####

writeLines(capture.output(sessionInfo()), safe_path(paste0("Session info_", Sys.Date(), ".txt")))
