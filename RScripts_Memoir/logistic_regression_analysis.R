# R Script: Logistic Regression Analysis - Pro-Palestinian Stance Over Time
# H1: Pro-Palestinian stance increases over time

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(lubridate)
library(broom)  # For tidy regression output
library(lmtest) # For robust standard errors
library(sandwich) # For robust standard errors

# Read the CSV file
data <- read_csv("/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv", 
                 locale = locale(encoding = "UTF-8"))

# Read the Gaza War Timeline
timeline <- read_delim("/home/jose/Documents/UNI/Mémoire/Data/Gaza War Timeline_ Operations & Discourse.csv", 
                       delim = ";",
                       locale = locale(encoding = "UTF-8"))

cat("Timeline events loaded:", nrow(timeline), "events\n")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Extract date and create time variables
data$date <- as.Date(data$`Post Created Date`, format = "%Y-%m-%d")
data$year_month <- format(data$date, "%Y-%m")
data$year_month_date <- as.Date(paste0(data$year_month, "-01"))

data$days_since_start <- as.numeric(data$date - min(data$date, na.rm = TRUE))
data$months_since_start <- as.numeric(difftime(data$date, min(data$date, na.rm = TRUE), units = "days")) / 30.44

# Create binary variable for Pro-Palestinian stance
data$is_pro_palestinian <- ifelse(data$predicted_category == "Pro-Palestinian", 1, 0)

# Parse timeline dates and create event markers
parse_timeline_dates <- function(period_str) {
  period_str <- trimws(period_str)
  if (grepl("–| - ", period_str)) {
    parts <- strsplit(period_str, "–| - ")[[1]]
    start_date <- tryCatch({
      as.Date(trimws(parts[1]), format = "%m/%d/%y")
    }, error = function(e) {
      as.Date(trimws(parts[1]), format = "%m/%d/%Y")
    })
    end_date <- tryCatch({
      as.Date(trimws(parts[2]), format = "%m/%d/%y")
    }, error = function(e) {
      as.Date(trimws(parts[2]), format = "%m/%d/%Y")
    })
    return(list(start = start_date, end = end_date, is_range = TRUE))
  } else {
    date_val <- tryCatch({
      as.Date(period_str, format = "%m/%d/%y")
    }, error = function(e) {
      tryCatch({
        as.Date(period_str, format = "%m/%d/%Y")
      }, error = function(e2) {
        as.Date(period_str, format = "%d/%m/%y")
      })
    })
    if (!is.na(date_val)) {
      return(list(start = date_val - 3, end = date_val + 3, is_range = FALSE))
    } else {
      return(list(start = NA, end = NA, is_range = FALSE))
    }
  }
}

# Process timeline events
timeline_events <- timeline %>%
  rowwise() %>%
  mutate(
    parsed = list(parse_timeline_dates(Period)),
    event_type = case_when(
      grepl("Ceasefire|Humanitarian Pause", `Major Military Operations / Events`, ignore.case = TRUE) ~ "Ceasefire",
      grepl("Operation|Offensive|Invasion|Attack|Strike|Raid", `Major Military Operations / Events`, ignore.case = TRUE) ~ "Military Operation",
      grepl("Evacuation", `Major Military Operations / Events`, ignore.case = TRUE) ~ "Evacuation",
      TRUE ~ "Other"
    )
  ) %>%
  ungroup()

# Create function to check if date is near a major event
is_near_major_event <- function(date_val, event_type_filter = NULL) {
  if (is.na(date_val)) return(FALSE)
  for (i in 1:nrow(timeline_events)) {
    period <- timeline_events$parsed[[i]]
    event_type <- timeline_events$event_type[i]
    
    if (!is.null(event_type_filter) && event_type != event_type_filter) next
    
    if (!is.null(period) && !is.na(period$start) && !is.na(period$end)) {
      if (date_val >= period$start && date_val <= period$end) {
        return(TRUE)
      }
    }
  }
  return(FALSE)
}

# Add timeline-based variables to data
data$near_military_operation <- sapply(data$date, function(d) is_near_major_event(d, "Military Operation"))
data$near_ceasefire <- sapply(data$date, function(d) is_near_major_event(d, "Ceasefire"))
data$near_major_event <- sapply(data$date, is_near_major_event)

# Convert to numeric
data$near_military_operation <- as.numeric(data$near_military_operation)
data$near_ceasefire <- as.numeric(data$near_ceasefire)
data$near_major_event <- as.numeric(data$near_major_event)

# Filter out missing values for key variables
data_clean <- data %>%
  filter(!is.na(date) & 
         !is.na(predicted_category) & 
         predicted_category != "")

cat("\nTimeline-based variables created:\n")
cat("  - Posts near military operations:", sum(data_clean$near_military_operation), "\n")
cat("  - Posts near ceasefire periods:", sum(data_clean$near_ceasefire), "\n")
cat("  - Posts near any major event:", sum(data_clean$near_major_event), "\n")

cat("Total observations for analysis:", nrow(data_clean), "\n")
cat("Pro-Palestinian posts:", sum(data_clean$is_pro_palestinian), "\n")
cat("Pro-Palestinian proportion:", round(mean(data_clean$is_pro_palestinian) * 100, 2), "%\n")

# ============================================================================
# LOGISTIC REGRESSION ANALYSIS
# ============================================================================

cat("\n=== LOGISTIC REGRESSION: Pro-Palestinian Stance Over Time ===\n\n")

# Model 1: Basic logistic regression at post level
model1 <- glm(is_pro_palestinian ~ months_since_start, 
              data = data_clean, 
              family = binomial(link = "logit"))
summary_model1 <- summary(model1)
cat("Model 1: Basic Logistic Regression (Post Level)\n")
print(summary_model1)

# Calculate odds ratio
odds_ratio <- exp(coef(model1)[2])
cat("\nOdds Ratio for time coefficient:\n")
cat("For each month, odds of Pro-Palestinian increase by", 
    round((odds_ratio - 1) * 100, 2), "%\n")
cat("Odds Ratio:", round(odds_ratio, 4), "\n")

# Model 2: Logistic regression with robust standard errors
robust_se1 <- vcovHC(model1, type = "HC3")
robust_test1 <- coeftest(model1, vcov = robust_se1)
cat("\nModel 2: Logistic Regression with Robust Standard Errors\n")
print(robust_test1)

model3 <- glm(is_pro_palestinian ~ months_since_start + I(months_since_start^2), 
              data = data_clean, 
              family = binomial(link = "logit"))
summary_model3 <- summary(model3)
cat("\nModel 3: Logistic Regression with Quadratic Time Trend\n")
print(summary_model3)

# Compare models
cat("\nModel Comparison (AIC):\n")
cat("Model 1 (Linear):", AIC(model1), "\n")
cat("Model 3 (Quadratic):", AIC(model3), "\n")
cat("Better model (lower AIC):", ifelse(AIC(model1) < AIC(model3), "Model 1 (Linear)", "Model 3 (Quadratic)"), "\n")

# ============================================================================
# AGGREGATED ANALYSIS (Monthly Level)
# ============================================================================

cat("\n=== AGGREGATED ANALYSIS (Monthly Level) ===\n\n")

# Aggregate to monthly level for time series analysis
monthly_data <- data_clean %>%
  group_by(year_month, year_month_date, months_since_start) %>%
  summarise(
    total_posts = n(),
    pro_palestinian_count = sum(is_pro_palestinian),
    pro_palestinian_prop = mean(is_pro_palestinian),
    .groups = "drop"
  ) %>%
  arrange(year_month_date)

model4 <- glm(cbind(pro_palestinian_count, total_posts - pro_palestinian_count) ~ months_since_start,
              data = monthly_data,
              family = binomial(link = "logit"))
summary_model4 <- summary(model4)
cat("Model 4: Logistic Regression on Aggregated Monthly Data\n")
print(summary_model4)

# Model 5: Logistic regression with heightened_violence as control
model5 <- glm(is_pro_palestinian ~ months_since_start + heightened_violence,
              data = data_clean,
              family = binomial(link = "logit"))
summary_model5 <- summary(model5)
cat("\nModel 5: Logistic Regression with Heightened Violence as Control\n")
print(summary_model5)

# Model 6: Logistic regression with interaction between time and heightened violence
model6 <- glm(is_pro_palestinian ~ months_since_start + heightened_violence + 
              months_since_start:heightened_violence,
              data = data_clean,
              family = binomial(link = "logit"))
summary_model6 <- summary(model6)
cat("\nModel 6: Logistic Regression with Time × Heightened Violence Interaction\n")
print(summary_model6)

# ============================================================================
# VISUALIZATION
# ============================================================================

# Prepare timeline events for plotting
timeline_plot_data <- timeline_events %>%
  rowwise() %>%
  mutate(
    event_date = ifelse(!is.null(parsed) && !is.na(parsed$start), parsed$start, NA),
    event_label = `Major Military Operations / Events`
  ) %>%
  ungroup() %>%
  filter(!is.na(event_date) & event_type %in% c("Military Operation", "Evacuation")) %>%
  arrange(event_date)

study_start <- min(monthly_data$year_month_date, na.rm = TRUE)
study_end <- max(monthly_data$year_month_date, na.rm = TRUE)
x_axis_limits <- c(study_start - 30, study_end + 30)

# Plot 1: Pro-Palestinian proportion over time with logistic curve and timeline events
p1 <- ggplot(monthly_data, aes(x = year_month_date, y = pro_palestinian_prop)) +
  geom_point(size = 2, color = "forestgreen", alpha = 0.7) +
  geom_smooth(method = "glm", 
              method.args = list(family = "binomial"),
              se = TRUE, 
              color = "red", 
              linetype = "dashed",
              aes(weight = total_posts)) +
  geom_vline(data = timeline_plot_data, 
             aes(xintercept = event_date), 
             color = "darkred", 
             linetype = "dotted", 
             alpha = 0.5,
             size = 0.5) +
  labs(
    title = "Pro-Palestinian Stance Over Time (Logistic Regression)",
    subtitle = paste("Time coefficient: β =", round(coef(model1)[2], 4), 
                     ", p =", round(summary_model1$coefficients[2,4], 4),
                     "\nOdds Ratio:", round(odds_ratio, 4),
                     " | Red dotted lines: Major military operations"),
    x = "Month",
    y = "Proportion of Pro-Palestinian Posts"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  )

ggsave("pro_palestinian_logistic_trend.png", plot = p1, width = 12, height = 6, dpi = 300)
cat("\nPlot saved to: pro_palestinian_logistic_trend.png\n")

pred_data <- data.frame(
  months_since_start = seq(min(data_clean$months_since_start, na.rm = TRUE),
                          max(data_clean$months_since_start, na.rm = TRUE),
                          length.out = 100)
)
pred_data$predicted_prob <- predict(model1, newdata = pred_data, type = "response")
pred_data$date <- min(data_clean$date, na.rm = TRUE) + (pred_data$months_since_start * 30.44)

p2 <- ggplot() +
  geom_point(data = monthly_data, 
             aes(x = year_month_date, y = pro_palestinian_prop, size = total_posts),
             color = "forestgreen", alpha = 0.6) +
  geom_line(data = pred_data, 
            aes(x = date, y = predicted_prob),
            color = "red", size = 1.2) +
  labs(
    title = "Predicted Probability of Pro-Palestinian Stance Over Time",
    subtitle = "Logistic Regression Model",
    x = "Month",
    y = "Probability of Pro-Palestinian Stance",
    size = "Number of Posts"
  ) +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  )

ggsave("pro_palestinian_predicted_probabilities.png", plot = p2, width = 12, height = 6, dpi = 300)
cat("Plot saved to: pro_palestinian_predicted_probabilities.png\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n\n=== SUMMARY OF LOGISTIC REGRESSION FINDINGS ===\n\n")
cat("H1: Pro-Palestinian Stance Increases Over Time\n")
cat("  - Time coefficient (months):", round(coef(model1)[2], 4), "\n")
cat("  - Standard Error:", round(summary_model1$coefficients[2,2], 4), "\n")
cat("  - Z-value:", round(summary_model1$coefficients[2,3], 4), "\n")
cat("  - P-value:", round(summary_model1$coefficients[2,4], 4), "\n")
cat("  - Odds Ratio:", round(odds_ratio, 4), "\n")
cat("  - Interpretation: Each month increases odds of Pro-Palestinian by", 
    round((odds_ratio - 1) * 100, 2), "%\n")
cat("  - Significance:", ifelse(summary_model1$coefficients[2,4] < 0.05, "YES (p < 0.05)", "NO (p >= 0.05)"), "\n\n")

cat("Model 5: With Heightened Violence as Control\n")
cat("  - Heightened violence effect:", round(coef(model5)[3], 4), 
    ", p =", round(summary_model5$coefficients[3,4], 4), "\n")
cat("  - Odds Ratio for heightened violence:", round(exp(coef(model5)[3]), 4), "\n\n")

cat("Model 6: Time × Heightened Violence Interaction\n")
cat("  - Interaction effect:", round(coef(model6)[4], 4), 
    ", p =", round(summary_model6$coefficients[4,4], 4), "\n")
cat("  - Interpretation: During heightened violence periods, the time trend", 
    ifelse(coef(model6)[4] > 0, "increases", "decreases"),
    "by", round(coef(model6)[4], 4), "units per month\n")

cat("\nAnalysis complete! Check generated plots and model summaries above.\n")

