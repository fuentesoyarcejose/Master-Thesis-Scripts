# R Script: Hypothesis Testing - Pro-Palestinian Stance and Humanitarian Framing
# H1: During periods of heightened violence, there is more engagement in posts 
#     with humanitarian framing (focus on violence or civilian suffering)

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

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Extract date and create time variables
data$date <- as.Date(data$`Post Created Date`, format = "%Y-%m-%d")
data$year_month <- format(data$date, "%Y-%m")
data$year_month_date <- as.Date(paste0(data$year_month, "-01"))

# Create numeric time variable (days since start, or months)
data$days_since_start <- as.numeric(data$date - min(data$date, na.rm = TRUE))
data$months_since_start <- as.numeric(difftime(data$date, min(data$date, na.rm = TRUE), units = "days")) / 30.44

# Create binary variable for Pro-Palestinian stance
data$is_pro_palestinian <- ifelse(data$predicted_category == "Pro-Palestinian", 1, 0)

data$humanitarian_framing <- (data$Care + data$Harm) / 2
data$is_humanitarian_frame <- ifelse(data$`highest score` %in% c("Care", "Harm"), 1, 0)

data$total_interactions <- as.numeric(gsub(",", "", data$`Total Interactions`))
data$total_interactions[is.na(data$total_interactions)] <- 0

data$likes <- as.numeric(data$Likes)
data$comments <- as.numeric(data$Comments)
data$shares <- as.numeric(data$Shares)
data$likes[is.na(data$likes)] <- 0
data$comments[is.na(data$comments)] <- 0
data$shares[is.na(data$shares)] <- 0

data$weighted_engagement <- data$likes + data$comments * 2 + data$shares * 3

if (!"heightened_violence" %in% names(data)) {
  stop("Error: 'heightened_violence' column not found in dataset. Please run add_heightened_violence_column.py first.")
}

data$heightened_violence <- as.numeric(data$heightened_violence)

cat("\nUsing heightened_violence column from dataset.\n")
cat("Posts during heightened violence periods:", sum(data$heightened_violence, na.rm = TRUE), "\n")
cat("Posts during normal periods:", sum(data$heightened_violence == 0, na.rm = TRUE), "\n")

# Filter out missing values for key variables
data_clean <- data %>%
  filter(!is.na(date) & 
         !is.na(predicted_category) & 
         predicted_category != "" &
         !is.na(total_interactions))

cat("Total observations for analysis:", nrow(data_clean), "\n")
cat("Pro-Palestinian posts:", sum(data_clean$is_pro_palestinian), "\n")
cat("Posts with humanitarian framing:", sum(data_clean$is_humanitarian_frame, na.rm = TRUE), "\n")
cat("Posts during heightened violence:", sum(data_clean$heightened_violence), "\n")

# ============================================================================
# HYPOTHESIS 1: Pro-Palestinian stance increases over time
# ============================================================================

cat("\n=== HYPOTHESIS 1: Pro-Palestinian Stance Increases Over Time ===\n\n")

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

# Model 1: Linear regression - Pro-Palestinian proportion over time
model1 <- lm(pro_palestinian_prop ~ months_since_start, data = monthly_data)
summary_model1 <- summary(model1)
cat("Model 1: Linear Trend\n")
print(summary_model1)

# Model 2: Time series with robust standard errors
model2 <- lm(pro_palestinian_prop ~ months_since_start, data = monthly_data)
robust_se2 <- vcovHC(model2, type = "HC3")
robust_test2 <- coeftest(model2, vcov = robust_se2)
cat("\nModel 2: Linear Trend with Robust Standard Errors\n")
print(robust_test2)

# ============================================================================
# HYPOTHESIS 2: More engagement in humanitarian framing during violent periods
# ============================================================================

cat("\n\n=== HYPOTHESIS 2: Engagement in Humanitarian Framing During Violent Periods ===\n\n")

# Filter for Pro-Palestinian posts only (as per your hypothesis)
pro_pal_data <- data_clean %>%
  filter(is_pro_palestinian == 1)

cat("Pro-Palestinian posts for analysis:", nrow(pro_pal_data), "\n")

pro_pal_data$log_engagement <- log(pro_pal_data$total_interactions + 1)
pro_pal_data$log_weighted_engagement <- log(pro_pal_data$weighted_engagement + 1)

model4 <- lm(log_engagement ~ is_humanitarian_frame + heightened_violence + 
             is_humanitarian_frame:heightened_violence,
             data = pro_pal_data)
summary_model4 <- summary(model4)
cat("Model 4: Engagement ~ Humanitarian Frame + Violent Period + Interaction\n")
print(summary_model4)

# Model 5: Using continuous humanitarian framing score
model5 <- lm(log_engagement ~ humanitarian_framing + heightened_violence + 
             humanitarian_framing:heightened_violence,
             data = pro_pal_data)
summary_model5 <- summary(model5)
cat("\nModel 5: Engagement ~ Humanitarian Score (continuous) + Violent Period + Interaction\n")
print(summary_model5)

library(MASS)
model6 <- glm.nb(total_interactions ~ is_humanitarian_frame + heightened_violence + 
                 is_humanitarian_frame:heightened_violence,
                 data = pro_pal_data)
summary_model6 <- summary(model6)
cat("\nModel 6: Negative Binomial Regression (for count data)\n")
print(summary_model6)

model7 <- lm(log_engagement ~ is_humanitarian_frame + heightened_violence + 
             is_humanitarian_frame:heightened_violence +
             Authority + Loyalty + Purity +
             months_since_start,
             data = pro_pal_data)
summary_model7 <- summary(model7)
cat("\nModel 7: With Controls (Other Moral Foundations + Time Trend)\n")
print(summary_model7)

# ============================================================================
# VISUALIZATION OF RESULTS
# ============================================================================

study_start <- min(monthly_data$year_month_date, na.rm = TRUE)
study_end <- max(monthly_data$year_month_date, na.rm = TRUE)
x_axis_limits <- c(study_start - 30, study_end + 30)

# Plot 1: Pro-Palestinian proportion over time with regression line
p1 <- ggplot(monthly_data, aes(x = year_month_date, y = pro_palestinian_prop)) +
  geom_point(size = 2, color = "forestgreen") +
  geom_smooth(method = "lm", se = TRUE, color = "red", linetype = "dashed") +
  labs(
    title = "Pro-Palestinian Stance Over Time",
    subtitle = paste("Linear trend: β =", round(coef(model1)[2], 4), 
                     ", p =", round(summary_model1$coefficients[2,4], 4)),
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

ggsave("pro_palestinian_trend.png", plot = p1, width = 12, height = 6, dpi = 300)

# Plot 2: Engagement by humanitarian framing and violent periods
p2 <- ggplot(pro_pal_data, 
             aes(x = factor(is_humanitarian_frame, 
                           labels = c("Other Frames", "Humanitarian Frame")),
                 y = log_engagement,
                 fill = factor(heightened_violence, 
                              labels = c("Normal Period", "Heightened Violence")))) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Engagement by Humanitarian Framing and Violent Periods",
    x = "Frame Type",
    y = "Log(Total Interactions)",
    fill = "Period"
  ) +
  theme_minimal()

ggsave("engagement_by_frame_and_violence.png", plot = p2, width = 10, height = 6, dpi = 300)

# ============================================================================
# SUMMARY TABLE
# ============================================================================

cat("\n\n=== SUMMARY OF KEY FINDINGS ===\n\n")
cat("H1: Pro-Palestinian Stance Over Time (Linear Regression)\n")
cat("  - Time coefficient (months):", round(coef(model1)[2], 4), "\n")
cat("  - P-value:", round(summary_model1$coefficients[2,4], 4), "\n")
cat("  - Interpretation: Each month increases proportion of Pro-Palestinian posts by", 
    round(coef(model1)[2] * 100, 2), "percentage points\n")
cat("  - Note: For logistic regression analysis, see logistic_regression_analysis.R\n\n")

cat("H2: Engagement in Humanitarian Framing During Violent Periods\n")
cat("  - Humanitarian frame effect:", round(coef(model4)[2], 4), "\n")
cat("  - Violent period effect:", round(coef(model4)[3], 4), "\n")
cat("  - Interaction effect (key test):", round(coef(model4)[4], 4), "\n")
cat("  - Interaction p-value:", round(summary_model4$coefficients[4,4], 4), "\n")
cat("  - Interpretation: During violent periods, humanitarian framing", 
    ifelse(coef(model4)[4] > 0, "increases", "decreases"),
    "engagement by", round(exp(coef(model4)[4]) - 1, 4) * 100, "%\n")

cat("\nAnalysis complete! Check generated plots and model summaries above.\n")

# ============================================================================
# ANNEX ANALYSIS: Using Daily Casualty Data (Deaths and Injuries)
# ============================================================================

cat("\n\n=== ANNEX ANALYSIS: Engagement and Humanitarian Framing Using Daily Casualty Data ===\n\n")

# Read the deaths and injuries data
casualties <- read_csv("/home/jose/Documents/UNI/Mémoire/Data/gaza_deaths_and_injuries.csv", 
                       locale = locale(encoding = "UTF-8"))

# Parse the report date
casualties$date <- as.Date(casualties$reportDate, format = "%Y-%m-%d")

# Create daily casualty variables
casualties <- casualties %>%
  mutate(
    daily_killed = killed,
    daily_injured = injured,
    daily_total_casualties = killed + injured,
    log_daily_killed = log(killed + 1),
    log_daily_injured = log(injured + 1),
    log_daily_total = log(daily_total_casualties + 1)
  ) %>%
  select(date, daily_killed, daily_injured, daily_total_casualties, 
         log_daily_killed, log_daily_injured, log_daily_total,
         totalKilled, totalInjured, childrenKilled, womenKilled)

cat("Casualty data date range:", min(casualties$date, na.rm = TRUE), "to", 
    max(casualties$date, na.rm = TRUE), "\n")
cat("Total casualty records:", nrow(casualties), "\n")

data_with_casualties <- data_clean %>%
  left_join(casualties, by = "date") %>%
  mutate(
    daily_killed = ifelse(is.na(daily_killed), 0, daily_killed),
    daily_injured = ifelse(is.na(daily_injured), 0, daily_injured),
    daily_total_casualties = ifelse(is.na(daily_total_casualties), 0, daily_total_casualties),
    log_daily_killed = ifelse(is.na(log_daily_killed), 0, log_daily_killed),
    log_daily_injured = ifelse(is.na(log_daily_injured), 0, log_daily_injured),
    log_daily_total = ifelse(is.na(log_daily_total), 0, log_daily_total)
  )

# Filter for Pro-Palestinian posts only
pro_pal_data_casualties <- data_with_casualties %>%
  filter(is_pro_palestinian == 1 & !is.na(log_engagement))

cat("Pro-Palestinian posts with casualty data:", nrow(pro_pal_data_casualties), "\n")

# ANNEX Model 1: Engagement ~ Humanitarian Frame + Daily Casualties + Interaction
annex_model1 <- lm(log_engagement ~ is_humanitarian_frame + log_daily_total + 
                    is_humanitarian_frame:log_daily_total,
                    data = pro_pal_data_casualties)
summary_annex1 <- summary(annex_model1)
cat("\nANNEX Model 1: Engagement ~ Humanitarian Frame + Daily Total Casualties (log) + Interaction\n")
print(summary_annex1)

# ANNEX Model 2: Using daily killed (more specific to violence)
annex_model2 <- lm(log_engagement ~ is_humanitarian_frame + log_daily_killed + 
                    is_humanitarian_frame:log_daily_killed,
                    data = pro_pal_data_casualties)
summary_annex2 <- summary(annex_model2)
cat("\nANNEX Model 2: Engagement ~ Humanitarian Frame + Daily Killed (log) + Interaction\n")
print(summary_annex2)

# ANNEX Model 3: Using continuous humanitarian framing score
annex_model3 <- lm(log_engagement ~ humanitarian_framing + log_daily_total + 
                    humanitarian_framing:log_daily_total,
                    data = pro_pal_data_casualties)
summary_annex3 <- summary(annex_model3)
cat("\nANNEX Model 3: Engagement ~ Humanitarian Score (continuous) + Daily Total Casualties (log) + Interaction\n")
print(summary_annex3)

# ANNEX Model 4: Negative binomial with casualty data
annex_model4 <- glm.nb(total_interactions ~ is_humanitarian_frame + log_daily_total + 
                       is_humanitarian_frame:log_daily_total,
                       data = pro_pal_data_casualties)
summary_annex4 <- summary(annex_model4)
cat("\nANNEX Model 4: Negative Binomial with Daily Total Casualties\n")
print(summary_annex4)

# ANNEX Model 5: With controls
annex_model5 <- lm(log_engagement ~ is_humanitarian_frame + log_daily_total + 
                    is_humanitarian_frame:log_daily_total +
                    Authority + Loyalty + Purity +  # Other moral foundations
                    months_since_start,  # Time trend
                    data = pro_pal_data_casualties)
summary_annex5 <- summary(annex_model5)
cat("\nANNEX Model 5: With Controls (Other Moral Foundations + Time Trend)\n")
print(summary_annex5)

pro_pal_data_casualties <- pro_pal_data_casualties %>%
  arrange(date) %>%
  mutate(
    lag1_daily_total = dplyr::lag(daily_total_casualties, 1),
    lag1_log_daily_total = dplyr::lag(log_daily_total, 1)
  )

annex_model6 <- lm(log_engagement ~ is_humanitarian_frame + lag1_log_daily_total + 
                    is_humanitarian_frame:lag1_log_daily_total,
                    data = pro_pal_data_casualties %>% filter(!is.na(lag1_log_daily_total)))
summary_annex6 <- summary(annex_model6)
cat("\nANNEX Model 6: Engagement ~ Humanitarian Frame + Lagged (1-day) Casualties + Interaction\n")
print(summary_annex6)

# ============================================================================
# ANNEX VISUALIZATIONS
# ============================================================================

pro_pal_data_casualties$casualty_category <- cut(pro_pal_data_casualties$daily_total_casualties,
                                                  breaks = c(-Inf, 0, 100, 500, 1000, Inf),
                                                  labels = c("No reports", "Low (1-100)", 
                                                           "Medium (101-500)", "High (501-1000)", 
                                                           "Very High (1000+)"))

p3 <- ggplot(pro_pal_data_casualties %>% filter(!is.na(casualty_category)), 
             aes(x = factor(is_humanitarian_frame, 
                           labels = c("Other Frames", "Humanitarian Frame")),
                 y = log_engagement,
                 fill = casualty_category)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    title = "Engagement by Humanitarian Framing and Daily Casualty Levels",
    x = "Frame Type",
    y = "Log(Total Interactions)",
    fill = "Daily Casualties"
  ) +
  theme_minimal()

ggsave("engagement_by_frame_and_casualties.png", plot = p3, width = 12, height = 6, dpi = 300)
cat("\nPlot saved to: engagement_by_frame_and_casualties.png\n")

# Plot: Scatter plot of engagement vs daily casualties by frame type
p4 <- ggplot(pro_pal_data_casualties %>% filter(daily_total_casualties > 0), 
             aes(x = log_daily_total, y = log_engagement, 
                 color = factor(is_humanitarian_frame, 
                               labels = c("Other Frames", "Humanitarian Frame")))) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Engagement vs Daily Casualties by Frame Type",
    x = "Log(Daily Total Casualties)",
    y = "Log(Total Interactions)",
    color = "Frame Type"
  ) +
  theme_minimal()

ggsave("engagement_vs_casualties_scatter.png", plot = p4, width = 12, height = 6, dpi = 300)
cat("Plot saved to: engagement_vs_casualties_scatter.png\n")

# ============================================================================
# ANNEX SUMMARY
# ============================================================================

cat("\n\n=== ANNEX ANALYSIS SUMMARY ===\n\n")
cat("H2 (Annex): Engagement in Humanitarian Framing During High Casualty Periods\n")
cat("  - Humanitarian frame effect:", round(coef(annex_model1)[2], 4), "\n")
cat("  - Daily casualties effect:", round(coef(annex_model1)[3], 4), "\n")
cat("  - Interaction effect (key test):", round(coef(annex_model1)[4], 4), "\n")
cat("  - Interaction p-value:", round(summary_annex1$coefficients[4,4], 4), "\n")
cat("  - Interpretation: For each log-unit increase in daily casualties, humanitarian framing", 
    ifelse(coef(annex_model1)[4] > 0, "increases", "decreases"),
    "engagement by", round(coef(annex_model1)[4], 4), "log units\n")

cat("\nAnnex analysis complete! Check generated plots and model summaries above.\n")

