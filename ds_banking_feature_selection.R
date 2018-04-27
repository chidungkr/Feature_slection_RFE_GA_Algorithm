

library(magrittr)
library(tidyverse)
breast <- read.csv("E:/R_project/w8_breast_cancer/data.csv")

# Bỏ biến không cần thiết: 
breast %<>% mutate(id = NULL, X = NULL)

# Hàm kiểm tra dữ liệu thiếu: 
rate_na <- function(x) {100*sum(is.na(x)) / length(x)}

# Sử dụng hàm: 
breast %>% summarise_all(rate_na)

# Kiêm tra kiểu dữ liệu cho biến mục tiêu: 
breast$diagnosis %>% class()

# Bộ dữ liệu nguyên bản: 
breast_origin <- breast

# Bộ dữ liệu bỏ đi các biến tương quan cao trên 0.8: 
input_df <- breast %>% select(-diagnosis)

library(caret)
tuong_quan_cao <- findCorrelation(cor(input_df), cutoff = 0.8)

# Số lượng các biến số sẽ bị loại: 
length(tuong_quan_cao)

# Loại các biến tương quan cao này và lấy lại biến mục tiêu: 
breast_remove_cor <- input_df %>% 
  select(-tuong_quan_cao) %>% 
  mutate(diagnosis = breast$diagnosis)

# Bộ dữ liệu bỏ tương quan cao và dữ liệu được chuẩn hóa 0 1: 
chuan_hoa_01 <- function(x) {(x - min(x)) / (max(x) - min(x))}
breast_scaled <- breast_remove_cor %>% mutate_if(is.numeric, chuan_hoa_01)


# Lựa chọn biến theo Recursive Feature Elimination: 

seeds <- vector(mode = "list", length = 26)
set.seed(29)
for(i in 1:25) seeds[[i]] <- sample.int(31, 30 + 1)
seeds[[26]] <- sample.int(29, 1)


set.seed(1)
control <- rfeControl(functions = rfFuncs, 
                      method = "repeatedcv", 
                      number = 5, 
                      repeats = 5, 
                      seeds = seeds, 
                      allowParallel = TRUE)


# Thiết lập tính toán song song: 
library(doParallel)
n_cores <- detectCores()
registerDoParallel(cores = n_cores - 1)

# Thực hiện thuật toán chọn biến: 

set.seed(1)
results <- rfe(breast %>% select(-diagnosis), 
               breast$diagnosis, 
               sizes = c(1:30), 
               rfeControl = control)


# Số lượng biến tối ưu: 
results$bestSubset

# Tên các biến đó là: 
var_names <- predictors(results)
var_names

# Hình ảnh hóa chất lượng dự báo khi số lượng biến được chọn thay đổi: 
theme_set(theme_minimal())
results$results %>% 
  select(Variables, Accuracy, Kappa) %>% 
  gather(a, b, -Variables) %>% 
  ggplot(aes(Variables, b)) + 
  geom_line() + 
  geom_point() + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL, 
       title = "Model Performance vs Number of Variables Selected Based on RFE Algorithm")


# Chỉ chọn các biến có danh sách ở trên: 
breast_rfe <- breast %>% select(c("diagnosis", var_names))

# Loại biến theo Genetic Algorithm (GA): 

set.seed(1)
control_ga <- gafsControl(functions = rfGA,
                          method = "repeatedcv",
                          number = 5,
                          repeats = 5,
                          seeds = 1:26, 
                          allowParallel = TRUE)

set.seed(1)
results_ga <- gafs(breast %>% select(-diagnosis),
                   breast$diagnosis,
                   gafsControl = control_ga)

# Các biến sẽ được sử dụng cho mô hình là: 
results_ga$ga$final



# Thiết lập Cross - Validation và các thống kê đánh giá mô hình: 

set.seed(1)
control <- trainControl(method = "repeatedcv", 
                        number = 5, 
                        repeats = 10, 
                        classProbs = TRUE, 
                        summaryFunction = multiClassSummary, 
                        allowParallel = TRUE)

# Huấn luyện mô hình SVM trên bốn bộ số liệu: 

set.seed(1)
svm_1 <- train(diagnosis ~ ., 
               data = breast_origin, 
               method = "svmLinear", 
               metric = "Accuracy",
               trControl = control)


set.seed(1)
svm_2 <- train(diagnosis ~ ., 
               data = breast_remove_cor, 
               method = "svmLinear", 
               metric = "Accuracy",
               trControl = control)


set.seed(1)
svm_3 <- train(diagnosis ~ ., 
               data = breast_scaled, 
               method = "svmLinear", 
               metric = "Accuracy",
               trControl = control)

set.seed(1)
svm_4 <- train(diagnosis ~ ., 
               data = breast_rfe, 
               method = "svmLinear", 
               metric = "Accuracy",
               trControl = control)


# So sánh bằng hình ảnh: 
all_df <- bind_rows(svm_1$results, 
                    svm_2$results, 
                    svm_3$results, 
                    svm_4$results)

all_df %<>% mutate(Method = c("All", "Cor", "CorS", "RFE"))

all_df %>% 
  select(AUC, Accuracy, Kappa, F1, Method) %>% 
  gather(a, b, -Method) %>% 
  ggplot(aes(Method, b, color = a)) + 
  geom_point(show.legend = FALSE, size = 3) + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL, 
       title = "Model Performance Based on Four Feature Selection Techniques", 
       subtitle = "Model Used: Support Vector Machine")



# Huấn luyện mô hình Random Forest trên bốn bộ số liệu: 

set.seed(1)
rf_1 <- train(diagnosis ~ ., 
              data = breast_origin, 
              method = "rf", 
              metric = "Accuracy",
              trControl = control)


set.seed(1)
rf_2 <- train(diagnosis ~ ., 
              data = breast_remove_cor, 
              method = "rf", 
              metric = "Accuracy",
              trControl = control)


set.seed(1)
rf_3 <- train(diagnosis ~ ., 
              data = breast_scaled, 
              method = "rf", 
              metric = "Accuracy",
              trControl = control)

set.seed(1)
rf_4 <- train(diagnosis ~ ., 
              data = breast_rfe, 
              method = "rf", 
              metric = "Accuracy",
              trControl = control)


# So sánh bằng hình ảnh: 

all_df <- bind_rows(rf_1$resample %>% select(-Resample) %>% summarise_all(funs(mean)), 
                    rf_2$resample %>% select(-Resample) %>% summarise_all(funs(mean)), 
                    rf_3$resample %>% select(-Resample) %>% summarise_all(funs(mean)), 
                    rf_4$resample %>% select(-Resample) %>% summarise_all(funs(mean)))


all_df %<>% mutate(Method = c("All", "Cor", "CorS", "RFE"))

all_df %>% 
  select(AUC, Accuracy, Kappa, F1, Method) %>% 
  gather(a, b, -Method) %>% 
  ggplot(aes(Method, b, color = a)) + 
  geom_point(show.legend = FALSE, size = 3) + 
  facet_wrap(~ a, scales = "free") + 
  labs(x = NULL, y = NULL, 
       title = "Model Performance Based on Four Feature Selection Techniques", 
       subtitle = "Model Used: Random Forest")