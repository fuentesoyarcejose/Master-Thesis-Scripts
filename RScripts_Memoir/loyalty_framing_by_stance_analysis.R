# R Script: Loyalty Framing During Heightened Violence by Stance
# Model: is_loyalty_frame ~ Time + heightened_violence * time
# Run separately for Pro-Palestinian, Pro-Israeli, Neutral, and All Posts

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

cat("=== LOYALTY FRAMING DURING HEIGHTENED VIOLENCE BY STANCE ===\n")
cat("Model: is_loyalty_frame ~ months_since_start + heightened_violence * months_since_start\n\n")

# Extract date
data$date <- as.Date(data$`Post Created Date`, format = "%Y-%m-%d")

# Create time variables
data$year_month <- format(data$date, "%Y-%m")
data$year_month_date <- as.Date(paste0(data$year_month, "-01"))
data$months_since_start <- as.numeric(difftime(data$date, min(data$date, na.rm = TRUE), units = "days")) / 30.44

# Check for heightened_violence column
if (!"heightened_violence" %in% names(data)) {
  stop("Error: 'heightened_violence' column not found in dataset. Please run add_heightened_violence_column.py first.")
}

# Ensure heightened_violence is numeric (0/1)
data$heightened_violence <- as.numeric(data$heightened_violence)

# Create stance variables
data$predicted_category <- as.character(data$predicted_category)

# Define loyalty framing
data$is_loyalty_frame <- ifelse(data$`highest score` == "Loyalty", 1, 0)

# Filter out missing values
data_clean <- data %>%
  filter(!is.na(date) & 
         !is.na(`highest score`) & 
         `highest score` != "" &
         !is.na(heightened_violence) &
         (predicted_category %in% c("Pro-Palestinian", "Pro-Israeli", "Neutral") | 
          !is.na(predicted_category)))

# Create separate datasets
data_pro_pal <- data_clean %>% filter(predicted_category == "Pro-Palestinian")
data_pro_isr <- data_clean %>% filter(predicted_category == "Pro-Israeli")
data_neutral <- data_clean %>% filter(predicted_category == "Neutral")
data_all <- data_clean  # All posts including neutral

cat("Dataset sizes:\n")
cat("  - Pro-Palestinian posts:", nrow(data_pro_pal), "\n")
cat("  - Pro-Israeli posts:", nrow(data_pro_isr), "\n")
cat("  - Neutral posts:", nrow(data_neutral), "\n")
cat("  - All posts (including neutral):", nrow(data_all), "\n\n")

# ============================================================================
# REGRESSION: Loyalty Framing ~ Time + Heightened Violence * Time
# ============================================================================

cat("=== REGRESSION MODEL ===\n")
cat("is_loyalty_frame ~ months_since_start + heightened_violence * months_since_start\n")
cat("This expands to: is_loyalty_frame ~ months_since_start + heightened_violence + \n")
cat("                  heightened_violence:months_since_start\n\n")

# Model for Pro-Palestinian posts
cat("--- Pro-Palestinian Posts ---\n")
model_pro_pal <- glm(is_loyalty_frame ~ months_since_start + 
                     heightened_violence * months_since_start, 
                     data = data_pro_pal, 
                     family = binomial(link = "logit"))
summary_model_pro_pal <- summary(model_pro_pal)
print(summary_model_pro_pal)

# Calculate odds ratios for Pro-Palestinian
odds_time_pro_pal <- exp(coef(model_pro_pal)[2])
odds_violence_pro_pal <- exp(coef(model_pro_pal)[3])
odds_interaction_pro_pal <- exp(coef(model_pro_pal)[4])

cat("\nOdds Ratios (Pro-Palestinian):\n")
cat("  - Time (months):", round(odds_time_pro_pal, 4), 
    " (p =", round(summary_model_pro_pal$coefficients[2,4], 6), ")\n")
cat("  - Heightened violence:", round(odds_violence_pro_pal, 4), 
    " (p =", round(summary_model_pro_pal$coefficients[3,4], 6), ")\n")
cat("  - Interaction (Violence × Time):", round(odds_interaction_pro_pal, 4), 
    " (p =", round(summary_model_pro_pal$coefficients[4,4], 6), ")\n\n")

# Model for Pro-Israeli posts
cat("--- Pro-Israeli Posts ---\n")
model_pro_isr <- glm(is_loyalty_frame ~ months_since_start + 
                     heightened_violence * months_since_start, 
                     data = data_pro_isr, 
                     family = binomial(link = "logit"))
summary_model_pro_isr <- summary(model_pro_isr)
print(summary_model_pro_isr)

# Calculate odds ratios for Pro-Israeli
odds_time_pro_isr <- exp(coef(model_pro_isr)[2])
odds_violence_pro_isr <- exp(coef(model_pro_isr)[3])
odds_interaction_pro_isr <- exp(coef(model_pro_isr)[4])

cat("\nOdds Ratios (Pro-Israeli):\n")
cat("  - Time (months):", round(odds_time_pro_isr, 4), 
    " (p =", round(summary_model_pro_isr$coefficients[2,4], 6), ")\n")
cat("  - Heightened violence:", round(odds_violence_pro_isr, 4), 
    " (p =", round(summary_model_pro_isr$coefficients[3,4], 6), ")\n")
cat("  - Interaction (Violence × Time):", round(odds_interaction_pro_isr, 4), 
    " (p =", round(summary_model_pro_isr$coefficients[4,4], 6), ")\n\n")

# Model for Neutral posts
cat("--- Neutral Posts ---\n")
model_neutral <- glm(is_loyalty_frame ~ months_since_start + 
                     heightened_violence * months_since_start, 
                     data = data_neutral, 
                     family = binomial(link = "logit"))
summary_model_neutral <- summary(model_neutral)
print(summary_model_neutral)

# Calculate odds ratios for Neutral
odds_time_neutral <- exp(coef(model_neutral)[2])
odds_violence_neutral <- exp(coef(model_neutral)[3])
odds_interaction_neutral <- exp(coef(model_neutral)[4])

cat("\nOdds Ratios (Neutral):\n")
cat("  - Time (months):", round(odds_time_neutral, 4), 
    " (p =", round(summary_model_neutral$coefficients[2,4], 6), ")\n")
cat("  - Heightened violence:", round(odds_violence_neutral, 4), 
    " (p =", round(summary_model_neutral$coefficients[3,4], 6), ")\n")
cat("  - Interaction (Violence × Time):", round(odds_interaction_neutral, 4), 
    " (p =", round(summary_model_neutral$coefficients[4,4], 6), ")\n\n")

# Model for All posts
cat("--- All Posts (Including Neutral) ---\n")
model_all <- glm(is_loyalty_frame ~ months_since_start + 
                 heightened_violence * months_since_start, 
                 data = data_all, 
                 family = binomial(link = "logit"))
summary_model_all <- summary(model_all)
print(summary_model_all)

# Calculate odds ratios for All posts
odds_time_all <- exp(coef(model_all)[2])
odds_violence_all <- exp(coef(model_all)[3])
odds_interaction_all <- exp(coef(model_all)[4])

cat("\nOdds Ratios (All Posts):\n")
cat("  - Time (months):", round(odds_time_all, 4), 
    " (p =", round(summary_model_all$coefficients[2,4], 6), ")\n")
cat("  - Heightened violence:", round(odds_violence_all, 4), 
    " (p =", round(summary_model_all$coefficients[3,4], 6), ")\n")
cat("  - Interaction (Violence × Time):", round(odds_interaction_all, 4), 
    " (p =", round(summary_model_all$coefficients[4,4], 6), ")\n\n")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

cat("=== COMPARISON TABLE ===\n\n")

comparison_table <- data.frame(
  Stance = c("Pro-Palestinian", "Pro-Israeli", "Neutral", "All Posts"),
  N = c(nrow(data_pro_pal), nrow(data_pro_isr), nrow(data_neutral), nrow(data_all)),
  Time_Coef = c(
    round(coef(model_pro_pal)[2], 4),
    round(coef(model_pro_isr)[2], 4),
    round(coef(model_neutral)[2], 4),
    round(coef(model_all)[2], 4)
  ),
  Time_OR = c(
    round(odds_time_pro_pal, 4),
    round(odds_time_pro_isr, 4),
    round(odds_time_neutral, 4),
    round(odds_time_all, 4)
  ),
  Time_P = c(
    round(summary_model_pro_pal$coefficients[2,4], 6),
    round(summary_model_pro_isr$coefficients[2,4], 6),
    round(summary_model_neutral$coefficients[2,4], 6),
    round(summary_model_all$coefficients[2,4], 6)
  ),
  Violence_Coef = c(
    round(coef(model_pro_pal)[3], 4),
    round(coef(model_pro_isr)[3], 4),
    round(coef(model_neutral)[3], 4),
    round(coef(model_all)[3], 4)
  ),
  Violence_OR = c(
    round(odds_violence_pro_pal, 4),
    round(odds_violence_pro_isr, 4),
    round(odds_violence_neutral, 4),
    round(odds_violence_all, 4)
  ),
  Violence_P = c(
    round(summary_model_pro_pal$coefficients[3,4], 6),
    round(summary_model_pro_isr$coefficients[3,4], 6),
    round(summary_model_neutral$coefficients[3,4], 6),
    round(summary_model_all$coefficients[3,4], 6)
  ),
  Interaction_Coef = c(
    round(coef(model_pro_pal)[4], 4),
    round(coef(model_pro_isr)[4], 4),
    round(coef(model_neutral)[4], 4),
    round(coef(model_all)[4], 4)
  ),
  Interaction_OR = c(
    round(odds_interaction_pro_pal, 4),
    round(odds_interaction_pro_isr, 4),
    round(odds_interaction_neutral, 4),
    round(odds_interaction_all, 4)
  ),
  Interaction_P = c(
    round(summary_model_pro_pal$coefficients[4,4], 6),
    round(summary_model_pro_isr$coefficients[4,4], 6),
    round(summary_model_neutral$coefficients[4,4], 6),
    round(summary_model_all$coefficients[4,4], 6)
  )
)

print(comparison_table)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n\n")

# Prepare prediction data for each stance
pred_data_pro_pal <- data.frame(
  months_since_start = seq(min(data_pro_pal$months_since_start, na.rm = TRUE),
                          max(data_pro_pal$months_since_start, na.rm = TRUE),
                          length.out = 100),
  heightened_violence = 0
)
pred_data_pro_pal$predicted_prob_normal <- predict(model_pro_pal, 
                                                   newdata = pred_data_pro_pal, 
                                                   type = "response")
pred_data_pro_pal$heightened_violence <- 1
pred_data_pro_pal$predicted_prob_violence <- predict(model_pro_pal, 
                                                      newdata = pred_data_pro_pal, 
                                                      type = "response")

pred_data_pro_isr <- data.frame(
  months_since_start = seq(min(data_pro_isr$months_since_start, na.rm = TRUE),
                          max(data_pro_isr$months_since_start, na.rm = TRUE),
                          length.out = 100),
  heightened_violence = 0
)
pred_data_pro_isr$predicted_prob_normal <- predict(model_pro_isr, 
                                                   newdata = pred_data_pro_isr, 
                                                   type = "response")
pred_data_pro_isr$heightened_violence <- 1
pred_data_pro_isr$predicted_prob_violence <- predict(model_pro_isr, 
                                                      newdata = pred_data_pro_isr, 
                                                      type = "response")

pred_data_neutral <- data.frame(
  months_since_start = seq(min(data_neutral$months_since_start, na.rm = TRUE),
                          max(data_neutral$months_since_start, na.rm = TRUE),
                          length.out = 100),
  heightened_violence = 0
)
pred_data_neutral$predicted_prob_normal <- predict(model_neutral, 
                                                   newdata = pred_data_neutral, 
                                                   type = "response")
pred_data_neutral$heightened_violence <- 1
pred_data_neutral$predicted_prob_violence <- predict(model_neutral, 
                                                     newdata = pred_data_neutral, 
                                                     type = "response")

pred_data_all <- data.frame(
  months_since_start = seq(min(data_all$months_since_start, na.rm = TRUE),
                          max(data_all$months_since_start, na.rm = TRUE),
                          length.out = 100),
  heightened_violence = 0
)
pred_data_all$predicted_prob_normal <- predict(model_all, 
                                               newdata = pred_data_all, 
                                               type = "response")
pred_data_all$heightened_violence <- 1
pred_data_all$predicted_prob_violence <- predict(model_all, 
                                                 newdata = pred_data_all, 
                                                 type = "response")

# Convert to dates
pred_data_pro_pal$date <- min(data_pro_pal$date, na.rm = TRUE) + (pred_data_pro_pal$months_since_start * 30.44)
pred_data_pro_isr$date <- min(data_pro_isr$date, na.rm = TRUE) + (pred_data_pro_isr$months_since_start * 30.44)
pred_data_neutral$date <- min(data_neutral$date, na.rm = TRUE) + (pred_data_neutral$months_since_start * 30.44)
pred_data_all$date <- min(data_all$date, na.rm = TRUE) + (pred_data_all$months_since_start * 30.44)

# Determine study period
study_start <- min(data_all$date, na.rm = TRUE)
study_end <- max(data_all$date, na.rm = TRUE)
x_axis_limits <- c(study_start - 30, study_end + 30)

# Plot 1: Faceted plot showing predicted probabilities for each stance
p1_data_normal <- rbind(
  data.frame(date = pred_data_pro_pal$date, prob = pred_data_pro_pal$predicted_prob_normal, 
             period = "Normal", stance = "Pro-Palestinian"),
  data.frame(date = pred_data_pro_isr$date, prob = pred_data_pro_isr$predicted_prob_normal, 
             period = "Normal", stance = "Pro-Israeli"),
  data.frame(date = pred_data_neutral$date, prob = pred_data_neutral$predicted_prob_normal, 
             period = "Normal", stance = "Neutral"),
  data.frame(date = pred_data_all$date, prob = pred_data_all$predicted_prob_normal, 
             period = "Normal", stance = "All Posts")
)

p1_data_violence <- rbind(
  data.frame(date = pred_data_pro_pal$date, prob = pred_data_pro_pal$predicted_prob_violence, 
             period = "Heightened Violence", stance = "Pro-Palestinian"),
  data.frame(date = pred_data_pro_isr$date, prob = pred_data_pro_isr$predicted_prob_violence, 
             period = "Heightened Violence", stance = "Pro-Israeli"),
  data.frame(date = pred_data_neutral$date, prob = pred_data_neutral$predicted_prob_violence, 
             period = "Heightened Violence", stance = "Neutral"),
  data.frame(date = pred_data_all$date, prob = pred_data_all$predicted_prob_violence, 
             period = "Heightened Violence", stance = "All Posts")
)

p1_data <- rbind(p1_data_normal, p1_data_violence)

p1 <- ggplot(p1_data, aes(x = date, y = prob, color = period, linetype = period)) +
  geom_line(size = 1.2) +
  facet_wrap(~ stance, ncol = 4) +
  scale_color_manual(values = c("Normal" = "gray50", "Heightened Violence" = "darkred")) +
  scale_linetype_manual(values = c("Normal" = "dashed", "Heightened Violence" = "solid")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  labs(
    title = "Predicted Probability of Loyalty Framing by Stance",
    subtitle = "Model: Time + Heightened Violence * Time",
    x = "Month",
    y = "Predicted Probability of Loyalty Framing",
    color = "Period",
    linetype = "Period"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave("loyalty_framing_by_stance.png", plot = p1, width = 18, height = 5, dpi = 300)
cat("Plot saved to: loyalty_framing_by_stance.png\n")

# Plot 2: Comparison of coefficients
coef_data <- data.frame(
  Stance = rep(c("Pro-Palestinian", "Pro-Israeli", "Neutral", "All Posts"), each = 3),
  Coefficient = c(
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction"
  ),
  Value = c(
    coef(model_pro_pal)[2], coef(model_pro_pal)[3], coef(model_pro_pal)[4],
    coef(model_pro_isr)[2], coef(model_pro_isr)[3], coef(model_pro_isr)[4],
    coef(model_neutral)[2], coef(model_neutral)[3], coef(model_neutral)[4],
    coef(model_all)[2], coef(model_all)[3], coef(model_all)[4]
  ),
  P_Value = c(
    summary_model_pro_pal$coefficients[2,4], summary_model_pro_pal$coefficients[3,4], summary_model_pro_pal$coefficients[4,4],
    summary_model_pro_isr$coefficients[2,4], summary_model_pro_isr$coefficients[3,4], summary_model_pro_isr$coefficients[4,4],
    summary_model_neutral$coefficients[2,4], summary_model_neutral$coefficients[3,4], summary_model_neutral$coefficients[4,4],
    summary_model_all$coefficients[2,4], summary_model_all$coefficients[3,4], summary_model_all$coefficients[4,4]
  )
)

p2 <- ggplot(coef_data, aes(x = Coefficient, y = Value, fill = Stance)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("Pro-Palestinian" = "forestgreen", 
                               "Pro-Israeli" = "royalblue",
                               "Neutral" = "darkorange",
                               "All Posts" = "gray50")) +
  labs(
    title = "Coefficients Comparison by Stance - Loyalty Framing",
    subtitle = "Logistic Regression: Loyalty Framing ~ Time + Heightened Violence * Time",
    x = "Coefficient",
    y = "Coefficient Value",
    fill = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("loyalty_framing_coefficients_comparison.png", plot = p2, width = 12, height = 6, dpi = 300)
cat("Plot saved to: loyalty_framing_coefficients_comparison.png\n")

# Plot 3: Odds Ratios comparison
or_data <- data.frame(
  Stance = rep(c("Pro-Palestinian", "Pro-Israeli", "Neutral", "All Posts"), each = 3),
  Coefficient = c(
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction",
    "Time", "Heightened Violence", "Interaction"
  ),
  Odds_Ratio = c(
    odds_time_pro_pal, odds_violence_pro_pal, odds_interaction_pro_pal,
    odds_time_pro_isr, odds_violence_pro_isr, odds_interaction_pro_isr,
    odds_time_neutral, odds_violence_neutral, odds_interaction_neutral,
    odds_time_all, odds_violence_all, odds_interaction_all
  )
)

p3 <- ggplot(or_data, aes(x = Coefficient, y = Odds_Ratio, fill = Stance)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("Pro-Palestinian" = "forestgreen", 
                               "Pro-Israeli" = "royalblue",
                               "Neutral" = "darkorange",
                               "All Posts" = "gray50")) +
  labs(
    title = "Odds Ratios Comparison by Stance - Loyalty Framing",
    subtitle = "Odds Ratio = 1 indicates no effect",
    x = "Coefficient",
    y = "Odds Ratio",
    fill = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("loyalty_framing_odds_ratios_comparison.png", plot = p3, width = 12, height = 6, dpi = 300)
cat("Plot saved to: loyalty_framing_odds_ratios_comparison.png\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n\n=== SUMMARY ===\n\n")
cat("Model: is_loyalty_frame ~ months_since_start + heightened_violence * months_since_start\n\n")

for (i in 1:nrow(comparison_table)) {
  stance_name <- comparison_table$Stance[i]
  cat(stance_name, ":\n", sep = "")
  cat("  - Time OR: ", comparison_table$Time_OR[i], 
      " (p = ", comparison_table$Time_P[i], ")\n", sep = "")
  cat("  - Heightened Violence OR: ", comparison_table$Violence_OR[i], 
      " (p = ", comparison_table$Violence_P[i], ")\n", sep = "")
  cat("  - Interaction OR: ", comparison_table$Interaction_OR[i], 
      " (p = ", comparison_table$Interaction_P[i], ")\n", sep = "")
  
  if (comparison_table$Time_P[i] < 0.05) {
    cat("  → Time significantly ", 
        ifelse(comparison_table$Time_OR[i] > 1, "increases", "decreases"),
        " loyalty framing\n", sep = "")
  }
  if (comparison_table$Violence_P[i] < 0.05) {
    cat("  → Heightened violence significantly ", 
        ifelse(comparison_table$Violence_OR[i] > 1, "increases", "decreases"),
        " loyalty framing\n", sep = "")
  }
  if (comparison_table$Interaction_P[i] < 0.05) {
    cat("  → The effect of violence on loyalty framing ", 
        ifelse(comparison_table$Interaction_OR[i] > 1, "increases", "decreases"),
        " over time\n", sep = "")
  }
  cat("\n")
}

cat("Analysis complete!\n")

