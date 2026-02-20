# R Script: Evolution of Framing and Engagement Relationship Over Time
# Research Question: How has the relationship between framing and user engagement evolved over time?
# 
# Filters:
# 1. Only accounts with followers (exclude Personal Profile)
# 2. Only accounts with 51+ posts (Intensive Participation category)

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(lubridate)
library(tidyr)  # For pivot_longer
library(broom)  # For tidy regression output
library(lmtest) # For robust standard errors
library(sandwich) # For robust standard errors

# Read the CSV file
data <- read_csv("/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv", 
                 locale = locale(encoding = "UTF-8"))

# ============================================================================
# DATA PREPARATION AND FILTERING
# ============================================================================

cat("=== FRAMING AND ENGAGEMENT EVOLUTION ANALYSIS ===\n\n")

# Extract date
data$date <- as.Date(data$`Post Created Date`, format = "%Y-%m-%d")

# Create time variables
data$year_month <- format(data$date, "%Y-%m")
data$year_month_date <- as.Date(paste0(data$year_month, "-01"))
data$months_since_start <- as.numeric(difftime(data$date, min(data$date, na.rm = TRUE), units = "days")) / 30.44

# Check column names for account type and followers
cat("Checking available columns...\n")
cat("Columns containing 'Type':", paste(grep("Type", names(data), value = TRUE, ignore.case = TRUE), collapse = ", "), "\n")
cat("Columns containing 'Follower':", paste(grep("Follower", names(data), value = TRUE, ignore.case = TRUE), collapse = ", "), "\n")
cat("Columns containing 'Facebook':", paste(grep("Facebook", names(data), value = TRUE, ignore.case = TRUE), collapse = ", "), "\n\n")

# Identify account type column (likely "Type" or similar)
account_type_col <- NULL
if ("Type" %in% names(data)) {
  account_type_col <- "Type"
} else if (any(grepl("Type", names(data), ignore.case = TRUE))) {
  account_type_col <- grep("Type", names(data), value = TRUE, ignore.case = TRUE)[1]
}

# Identify followers column
followers_col <- NULL
if ("Followers at Posting" %in% names(data)) {
  followers_col <- "Followers at Posting"
} else if (any(grepl("Follower", names(data), ignore.case = TRUE))) {
  followers_col <- grep("Follower", names(data), value = TRUE, ignore.case = TRUE)[1]
}

# Identify Facebook ID column (for counting posts per account)
facebook_id_col <- NULL
if ("Facebook Id" %in% names(data)) {
  facebook_id_col <- "Facebook Id"
} else if (any(grepl("Facebook.*Id", names(data), ignore.case = TRUE))) {
  facebook_id_col <- grep("Facebook.*Id", names(data), value = TRUE, ignore.case = TRUE)[1]
} else if (any(grepl("Id", names(data), ignore.case = TRUE))) {
  facebook_id_col <- grep("Id", names(data), value = TRUE, ignore.case = TRUE)[1]
}

cat("Using columns:\n")
cat("  - Account Type:", ifelse(is.null(account_type_col), "NOT FOUND", account_type_col), "\n")
cat("  - Followers:", ifelse(is.null(followers_col), "NOT FOUND", followers_col), "\n")
cat("  - Facebook ID:", ifelse(is.null(facebook_id_col), "NOT FOUND", facebook_id_col), "\n\n")

# Check account type values
if (!is.null(account_type_col)) {
  cat("Account Type distribution:\n")
  print(table(data[[account_type_col]], useNA = "ifany"))
  cat("\n")
}

# Calculate posts per account
if (!is.null(facebook_id_col)) {
  cat("Calculating posts per account...\n")
  account_post_counts <- data %>%
    group_by(!!sym(facebook_id_col)) %>%
    summarise(
      post_count = n(),
      .groups = "drop"
    )
  
  data <- data %>%
    left_join(account_post_counts, by = facebook_id_col)
  
  cat("Post count distribution:\n")
  print(summary(account_post_counts$post_count))
  cat("\n")
} else {
  cat("Warning: Cannot identify Facebook ID column. Post count filtering will be skipped.\n")
  data$post_count <- 1  # Placeholder
}

# FILTER 1: Exclude Personal Profile accounts (keep only accounts with followers)
# Strategy: Filter by having followers > 0 (Personal Profiles typically have 0 or very few followers)
if (!is.null(followers_col)) {
  cat("Filter 1: Filtering to accounts with followers (excluding Personal Profiles)...\n")
  cat("  Before filter:", nrow(data), "posts\n")
  
  # Convert followers to numeric
  data$followers_numeric <- as.numeric(gsub(",", "", data[[followers_col]]))
  data$followers_numeric[is.na(data$followers_numeric)] <- 0
  
  # Filter: keep only posts from accounts with followers > 0
  # Personal Profiles typically have 0 followers or are not public
  data_filtered <- data %>%
    filter(followers_numeric > 0)
  
  cat("  After filter:", nrow(data_filtered), "posts\n")
  cat("  Removed:", nrow(data) - nrow(data_filtered), "posts\n")
  cat("  Accounts with followers:", length(unique(data_filtered[[facebook_id_col]])), "\n\n")
} else if (!is.null(account_type_col)) {
  cat("Filter 1: Excluding Personal Profile accounts by Type column...\n")
  cat("  Before filter:", nrow(data), "posts\n")
  
  # Check what account types exist
  account_types <- unique(data[[account_type_col]])
  cat("  Account types found:", paste(account_types[!is.na(account_types) & account_types != ""], collapse = ", "), "\n")
  
  # Exclude "Personal Profile" if it exists
  data_filtered <- data %>%
    filter(!is.na(!!sym(account_type_col)) &
           !!sym(account_type_col) != "Personal Profile" &
           !!sym(account_type_col) != "")
  
  cat("  After filter:", nrow(data_filtered), "posts\n")
  cat("  Removed:", nrow(data) - nrow(data_filtered), "posts\n\n")
} else {
  cat("Warning: Neither Followers nor Account Type column found. Skipping account type filter.\n")
  data_filtered <- data
}

# FILTER 2: Only accounts with 51+ posts (Intensive Participation)
cat("Filter 2: Keeping only accounts with 51+ posts (Intensive Participation)...\n")
cat("  Before filter:", nrow(data_filtered), "posts\n")

data_filtered <- data_filtered %>%
  filter(post_count >= 51)

cat("  After filter:", nrow(data_filtered), "posts\n")
cat("  Removed:", nrow(data) - nrow(data_filtered), "posts\n")
cat("  Unique accounts:", length(unique(data_filtered[[facebook_id_col]])), "\n\n")

# Create engagement variable
data_filtered$total_interactions <- as.numeric(gsub(",", "", data_filtered$`Total Interactions`))
data_filtered$total_interactions[is.na(data_filtered$total_interactions)] <- 0

# Log transform engagement (to handle skewed distribution)
data_filtered$log_engagement <- log(data_filtered$total_interactions + 1)

# Define moral frames
data_filtered$is_humanitarian_frame <- ifelse(data_filtered$`highest score` %in% c("Care", "Harm"), 1, 0)
data_filtered$is_group_binding_frame <- ifelse(data_filtered$`highest score` %in% c("Loyalty", "Authority"), 1, 0)
data_filtered$is_loyalty_frame <- ifelse(data_filtered$`highest score` == "Loyalty", 1, 0)
data_filtered$is_authority_frame <- ifelse(data_filtered$`highest score` == "Authority", 1, 0)

# Create stance variable
data_filtered$predicted_category <- as.character(data_filtered$predicted_category)
data_filtered$stance <- factor(data_filtered$predicted_category, 
                               levels = c("Neutral", "Pro-Palestinian", "Pro-Israeli"))

# Create binary variable for Pro-Palestinian stance
data_filtered$is_pro_palestinian <- ifelse(data_filtered$predicted_category == "Pro-Palestinian", 1, 0)

# Final data cleaning
data_clean <- data_filtered %>%
  filter(!is.na(date) & 
         !is.na(`highest score`) & 
         `highest score` != "" &
         !is.na(total_interactions))

cat("Final dataset for analysis:\n")
cat("  - Total posts:", nrow(data_clean), "\n")
cat("  - Unique accounts:", length(unique(data_clean[[facebook_id_col]])), "\n")
cat("  - Date range:", min(data_clean$date, na.rm = TRUE), "to", max(data_clean$date, na.rm = TRUE), "\n")
cat("  - Mean engagement:", round(mean(data_clean$total_interactions, na.rm = TRUE), 2), "\n")
cat("  - Median engagement:", round(median(data_clean$total_interactions, na.rm = TRUE), 2), "\n\n")

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================

cat("=== DESCRIPTIVE STATISTICS ===\n\n")

# Engagement by framing type
engagement_by_framing <- data_clean %>%
  group_by(`highest score`) %>%
  summarise(
    n = n(),
    mean_engagement = mean(total_interactions, na.rm = TRUE),
    median_engagement = median(total_interactions, na.rm = TRUE),
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_engagement))

cat("Engagement by Moral Frame:\n")
print(engagement_by_framing)

# Engagement by stance
engagement_by_stance <- data_clean %>%
  group_by(predicted_category) %>%
  summarise(
    n = n(),
    mean_engagement = mean(total_interactions, na.rm = TRUE),
    median_engagement = median(total_interactions, na.rm = TRUE),
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nEngagement by Stance:\n")
print(engagement_by_stance)

# ============================================================================
# RESEARCH QUESTION: Does Engagement Lead to Pro-Palestinian Stance and/or Humanitarian Framing?
# ============================================================================

cat("\n=== RESEARCH QUESTION: Does Engagement Lead to Pro-Palestinian Stance and/or Humanitarian Framing? ===\n\n")

if (!is.null(facebook_id_col)) {
  cat("Creating lagged engagement variables...\n")
  
  data_clean <- data_clean %>%
    arrange(!!sym(facebook_id_col), date) %>%
    group_by(!!sym(facebook_id_col)) %>%
    mutate(
      lag1_engagement = dplyr::lag(total_interactions, 1),
      lag1_log_engagement = dplyr::lag(log_engagement, 1),
      lag_avg_engagement = (dplyr::lag(total_interactions, 1) + 
                            dplyr::lag(total_interactions, 2) + 
                            dplyr::lag(total_interactions, 3)) / 3,
      lag_avg_log_engagement = (dplyr::lag(log_engagement, 1) + 
                                dplyr::lag(log_engagement, 2) + 
                                dplyr::lag(log_engagement, 3)) / 3,
      lag_max_engagement = pmax(dplyr::lag(total_interactions, 1),
                                dplyr::lag(total_interactions, 2),
                                dplyr::lag(total_interactions, 3),
                                dplyr::lag(total_interactions, 4),
                                dplyr::lag(total_interactions, 5),
                                na.rm = TRUE)
    ) %>%
    ungroup()
  
  cat("Lagged engagement variables created.\n")
  cat("Posts with lagged engagement data:", sum(!is.na(data_clean$lag1_engagement)), "\n\n")
} else {
  cat("Warning: Cannot create lagged variables without Facebook ID.\n")
  data_clean$lag1_engagement <- NA
  data_clean$lag1_log_engagement <- NA
  data_clean$lag_avg_engagement <- NA
  data_clean$lag_avg_log_engagement <- NA
  data_clean$lag_max_engagement <- NA
}

data_clean$high_engagement <- ifelse(data_clean$total_interactions > 
                                     median(data_clean$total_interactions, na.rm = TRUE), 1, 0)

# ============================================================================
# REGRESSION MODELS: Does Engagement Lead to Pro-Palestinian Stance?
# ============================================================================

cat("=== MODELS: Does Engagement Lead to Pro-Palestinian Stance? ===\n\n")

cat("--- Model A1: Pro-Palestinian Stance ~ Current Engagement + Time ---\n")
model_a1 <- glm(is_pro_palestinian ~ log_engagement + months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_a1 <- summary(model_a1)
print(summary_model_a1)

odds_engagement_a1 <- exp(coef(model_a1)[2])
cat("\nOdds Ratio for engagement:", round(odds_engagement_a1, 4), "\n")
cat("Interpretation: Each unit increase in log(engagement) increases odds of Pro-Palestinian by", 
    round((odds_engagement_a1 - 1) * 100, 2), "%\n\n")

cat("--- Model A2: Pro-Palestinian Stance ~ Lagged Engagement (Previous Post) + Time ---\n")
model_a2 <- glm(is_pro_palestinian ~ lag1_log_engagement + months_since_start,
                data = data_clean %>% filter(!is.na(lag1_log_engagement)),
                family = binomial(link = "logit"))
summary_model_a2 <- summary(model_a2)
print(summary_model_a2)

if (length(coef(model_a2)) > 1) {
  odds_engagement_a2 <- exp(coef(model_a2)[2])
  cat("\nOdds Ratio for lagged engagement:", round(odds_engagement_a2, 4), "\n")
  cat("Interpretation: High engagement in previous post increases odds of current Pro-Palestinian by", 
      round((odds_engagement_a2 - 1) * 100, 2), "%\n\n")
}

cat("--- Model A3: Pro-Palestinian Stance ~ Average Lagged Engagement (Previous 3 Posts) + Time ---\n")
model_a3 <- glm(is_pro_palestinian ~ lag_avg_log_engagement + months_since_start,
                data = data_clean %>% filter(!is.na(lag_avg_log_engagement)),
                family = binomial(link = "logit"))
summary_model_a3 <- summary(model_a3)
print(summary_model_a3)

if (length(coef(model_a3)) > 1) {
  odds_engagement_a3 <- exp(coef(model_a3)[2])
  cat("\nOdds Ratio for average lagged engagement:", round(odds_engagement_a3, 4), "\n\n")
}

cat("--- Model A4: Pro-Palestinian Stance ~ Engagement × Time (Interaction) ---\n")
model_a4 <- glm(is_pro_palestinian ~ log_engagement * months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_a4 <- summary(model_a4)
print(summary_model_a4)

# ============================================================================
# REGRESSION MODELS: Does Engagement Lead to Humanitarian Framing?
# ============================================================================

cat("\n=== MODELS: Does Engagement Lead to Humanitarian Framing? ===\n\n")

cat("--- Model B1: Humanitarian Framing ~ Current Engagement + Time ---\n")
model_b1 <- glm(is_humanitarian_frame ~ log_engagement + months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_b1 <- summary(model_b1)
print(summary_model_b1)

odds_engagement_b1 <- exp(coef(model_b1)[2])
cat("\nOdds Ratio for engagement:", round(odds_engagement_b1, 4), "\n")
cat("Interpretation: Each unit increase in log(engagement) increases odds of humanitarian framing by", 
    round((odds_engagement_b1 - 1) * 100, 2), "%\n\n")

cat("--- Model B2: Humanitarian Framing ~ Lagged Engagement (Previous Post) + Time ---\n")
model_b2 <- glm(is_humanitarian_frame ~ lag1_log_engagement + months_since_start,
                data = data_clean %>% filter(!is.na(lag1_log_engagement)),
                family = binomial(link = "logit"))
summary_model_b2 <- summary(model_b2)
print(summary_model_b2)

if (length(coef(model_b2)) > 1) {
  odds_engagement_b2 <- exp(coef(model_b2)[2])
  cat("\nOdds Ratio for lagged engagement:", round(odds_engagement_b2, 4), "\n")
  cat("Interpretation: High engagement in previous post increases odds of current humanitarian framing by", 
      round((odds_engagement_b2 - 1) * 100, 2), "%\n\n")
}

cat("--- Model B3: Humanitarian Framing ~ Average Lagged Engagement (Previous 3 Posts) + Time ---\n")
model_b3 <- glm(is_humanitarian_frame ~ lag_avg_log_engagement + months_since_start,
                data = data_clean %>% filter(!is.na(lag_avg_log_engagement)),
                family = binomial(link = "logit"))
summary_model_b3 <- summary(model_b3)
print(summary_model_b3)

if (length(coef(model_b3)) > 1) {
  odds_engagement_b3 <- exp(coef(model_b3)[2])
  cat("\nOdds Ratio for average lagged engagement:", round(odds_engagement_b3, 4), "\n\n")
}

cat("--- Model B4: Humanitarian Framing ~ Engagement × Time (Interaction) ---\n")
model_b4 <- glm(is_humanitarian_frame ~ log_engagement * months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_b4 <- summary(model_b4)
print(summary_model_b4)

# ============================================================================
# COMBINED MODELS: Engagement Leading to Both Pro-Palestinian AND Humanitarian Framing
# ============================================================================

cat("\n=== COMBINED MODELS: Engagement Leading to Pro-Palestinian AND Humanitarian Framing ===\n\n")

cat("--- Model C1: Pro-Palestinian Stance ~ Engagement + Humanitarian Framing + Engagement × Humanitarian ---\n")
model_c1 <- glm(is_pro_palestinian ~ log_engagement + is_humanitarian_frame + 
                log_engagement:is_humanitarian_frame + months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_c1 <- summary(model_c1)
print(summary_model_c1)

cat("--- Model C2: Pro-Palestinian Stance ~ Lagged Engagement + Current Humanitarian Framing + Time ---\n")
model_c2 <- glm(is_pro_palestinian ~ lag1_log_engagement + is_humanitarian_frame + months_since_start,
                data = data_clean %>% filter(!is.na(lag1_log_engagement)),
                family = binomial(link = "logit"))
summary_model_c2 <- summary(model_c2)
print(summary_model_c2)

cat("\n--- Model C3: Pro-Palestinian Stance ~ High Engagement × Time (Does engagement effect change over time?) ---\n")
model_c3 <- glm(is_pro_palestinian ~ high_engagement * months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_c3 <- summary(model_c3)
print(summary_model_c3)

cat("\n--- Model C4: Humanitarian Framing ~ High Engagement × Time (Does engagement effect change over time?) ---\n")
model_c4 <- glm(is_humanitarian_frame ~ high_engagement * months_since_start,
                data = data_clean,
                family = binomial(link = "logit"))
summary_model_c4 <- summary(model_c4)
print(summary_model_c4)

# ============================================================================
# REGRESSION MODELS: Engagement ~ Framing + Time + Interactions (Original Direction)
# ============================================================================

cat("\n=== REGRESSION MODELS: Engagement ~ Framing + Time (Original Direction) ===\n\n")

cat("--- Model 1: Log(Engagement) ~ Humanitarian Framing + Time + Stance + Interactions ---\n")
model1 <- lm(log_engagement ~ is_humanitarian_frame + months_since_start + stance +
             is_humanitarian_frame:months_since_start +
             is_humanitarian_frame:stance +
             months_since_start:stance,
             data = data_clean)
summary_model1 <- summary(model1)
print(summary_model1)

robust_se1 <- vcovHC(model1, type = "HC3")
robust_test1 <- coeftest(model1, vcov = robust_se1)
cat("\nModel 1 with Robust Standard Errors:\n")
print(robust_test1)

cat("\n--- Model 2: Log(Engagement) ~ Group-Binding Framing + Time + Stance + Interactions ---\n")
model2 <- lm(log_engagement ~ is_group_binding_frame + months_since_start + stance +
             is_group_binding_frame:months_since_start +
             is_group_binding_frame:stance +
             months_since_start:stance,
             data = data_clean)
summary_model2 <- summary(model2)
print(summary_model2)

robust_se2 <- vcovHC(model2, type = "HC3")
robust_test2 <- coeftest(model2, vcov = robust_se2)
cat("\nModel 2 with Robust Standard Errors:\n")
print(robust_test2)

cat("\n--- Model 3: Log(Engagement) ~ Humanitarian Framing + Group-Binding Framing + Time + Stance + All Interactions ---\n")
model3 <- lm(log_engagement ~ is_humanitarian_frame + is_group_binding_frame + 
             months_since_start + stance +
             is_humanitarian_frame:months_since_start +
             is_group_binding_frame:months_since_start +
             is_humanitarian_frame:stance +
             is_group_binding_frame:stance +
             is_humanitarian_frame:stance:months_since_start +
             is_group_binding_frame:stance:months_since_start,
             data = data_clean)
summary_model3 <- summary(model3)
print(summary_model3)

cat("\n--- Model 4: Log(Engagement) ~ Humanitarian Framing × Time × Stance (Full 3-way interaction) ---\n")
model4 <- lm(log_engagement ~ is_humanitarian_frame * months_since_start * stance,
             data = data_clean)
summary_model4 <- summary(model4)
print(summary_model4)

cat("\n--- Model 5: Log(Engagement) ~ Group-Binding Framing × Time × Stance (Full 3-way interaction) ---\n")
model5 <- lm(log_engagement ~ is_group_binding_frame * months_since_start * stance,
             data = data_clean)
summary_model5 <- summary(model5)
print(summary_model5)

# ============================================================================
# STANCE EVOLUTION ANALYSIS: Do accounts shift to Pro-Palestinian?
# ============================================================================

cat("\n=== STANCE EVOLUTION ANALYSIS ===\n\n")

if (!is.null(facebook_id_col)) {
  cat("Analyzing stance evolution at account level...\n")
  
  account_stance_monthly <- data_clean %>%
    group_by(!!sym(facebook_id_col), year_month, year_month_date) %>%
    summarise(
      total_posts = n(),
      pro_pal_posts = sum(is_pro_palestinian),
      pro_pal_prop = mean(is_pro_palestinian),
      mean_engagement = mean(total_interactions, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(!!sym(facebook_id_col), year_month_date)
  
  account_stance_evolution <- data_clean %>%
    group_by(!!sym(facebook_id_col)) %>%
    arrange(date) %>%
    summarise(
      first_date = min(date, na.rm = TRUE),
      last_date = max(date, na.rm = TRUE),
      first_stance = first(predicted_category),
      last_stance = last(predicted_category),
      total_posts = n(),
      pro_pal_prop_first_month = mean(is_pro_palestinian[date == min(date, na.rm = TRUE)], na.rm = TRUE),
      pro_pal_prop_last_month = mean(is_pro_palestinian[date == max(date, na.rm = TRUE)], na.rm = TRUE),
      pro_pal_prop_overall = mean(is_pro_palestinian, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      stance_change = case_when(
        first_stance != "Pro-Palestinian" & last_stance == "Pro-Palestinian" ~ "Changed to Pro-Palestinian",
        first_stance == "Pro-Palestinian" & last_stance != "Pro-Palestinian" ~ "Changed from Pro-Palestinian",
        first_stance == "Pro-Palestinian" & last_stance == "Pro-Palestinian" ~ "Stayed Pro-Palestinian",
        TRUE ~ "Stayed Non-Pro-Palestinian"
      ),
      pro_pal_increase = pro_pal_prop_last_month - pro_pal_prop_first_month
    )
  
  cat("Stance evolution summary:\n")
  print(table(account_stance_evolution$stance_change))
  cat("\n")
  
  cat("Accounts that changed to Pro-Palestinian:", 
      sum(account_stance_evolution$stance_change == "Changed to Pro-Palestinian"), "\n")
  cat("Accounts that increased Pro-Palestinian proportion:", 
      sum(account_stance_evolution$pro_pal_increase > 0, na.rm = TRUE), "\n")
  cat("Mean Pro-Palestinian proportion increase:", 
      round(mean(account_stance_evolution$pro_pal_increase, na.rm = TRUE), 4), "\n\n")
}

cat("--- Model: Pro-Palestinian Stance ~ Time (for filtered accounts) ---\n")
model_stance <- glm(is_pro_palestinian ~ months_since_start, 
                    data = data_clean, 
                    family = binomial(link = "logit"))
summary_model_stance <- summary(model_stance)
print(summary_model_stance)

odds_ratio_stance <- exp(coef(model_stance)[2])
cat("\nOdds Ratio for time:", round(odds_ratio_stance, 4), "\n")
cat("Interpretation: Each month increases odds of Pro-Palestinian by", 
    round((odds_ratio_stance - 1) * 100, 2), "%\n\n")

# ============================================================================
# TIME-SERIES ANALYSIS: Monthly Evolution
# ============================================================================

cat("\n=== TIME-SERIES ANALYSIS: Monthly Evolution ===\n\n")

# Aggregate by month, framing, and stance
monthly_engagement <- data_clean %>%
  group_by(year_month, year_month_date, is_humanitarian_frame, is_group_binding_frame, predicted_category) %>%
  summarise(
    n_posts = n(),
    mean_engagement = mean(total_interactions, na.rm = TRUE),
    median_engagement = median(total_interactions, na.rm = TRUE),
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    pro_pal_prop = mean(is_pro_palestinian, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(year_month_date)

cat("Monthly engagement by framing and stance (sample):\n")
print(head(monthly_engagement, 20))

# Monthly stance proportions
monthly_stance <- data_clean %>%
  group_by(year_month, year_month_date) %>%
  summarise(
    total_posts = n(),
    pro_pal_prop = mean(is_pro_palestinian, na.rm = TRUE),
    pro_isr_prop = mean(ifelse(predicted_category == "Pro-Israeli", 1, 0), na.rm = TRUE),
    neutral_prop = mean(ifelse(predicted_category == "Neutral", 1, 0), na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(year_month_date)

cat("\nMonthly stance proportions:\n")
print(tail(monthly_stance, 10))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

cat("\n=== CREATING VISUALIZATIONS ===\n\n")

study_start <- min(data_clean$year_month_date, na.rm = TRUE)
study_end <- max(data_clean$year_month_date, na.rm = TRUE)
x_axis_limits <- c(study_start - 30, study_end + 30)

# Plot 1: Engagement over time by humanitarian framing and stance
monthly_humanitarian_stance <- data_clean %>%
  group_by(year_month_date, is_humanitarian_frame, predicted_category) %>%
  summarise(
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    se_log_engagement = sd(log_engagement, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 3) %>%  # Only show months with at least 3 posts
  mutate(
    frame_type = ifelse(is_humanitarian_frame == 1, "Humanitarian Frame", "Other Frames")
  )

p1 <- ggplot(monthly_humanitarian_stance, 
             aes(x = year_month_date, y = mean_log_engagement, 
                 color = interaction(predicted_category, frame_type), 
                 linetype = frame_type,
                 group = interaction(predicted_category, frame_type))) +
  geom_line(size = 1.2) +
  geom_point(size = 2, alpha = 0.7) +
  scale_color_manual(values = c(
    "Pro-Palestinian.Humanitarian Frame" = "darkred",
    "Pro-Palestinian.Other Frames" = "lightcoral",
    "Pro-Israeli.Humanitarian Frame" = "darkblue",
    "Pro-Israeli.Other Frames" = "lightblue",
    "Neutral.Humanitarian Frame" = "darkorange",
    "Neutral.Other Frames" = "orange"
  )) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  labs(
    title = "Evolution of Engagement by Humanitarian Framing and Stance Over Time",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Month",
    y = "Mean Log(Total Interactions)",
    color = "Stance × Frame",
    linetype = "Frame Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("engagement_evolution_humanitarian_framing_stance.png", plot = p1, width = 16, height = 6, dpi = 300)
cat("Plot saved to: engagement_evolution_humanitarian_framing_stance.png\n")

# Plot 2: Engagement over time by group-binding framing and stance
monthly_group_binding_stance <- data_clean %>%
  group_by(year_month_date, is_group_binding_frame, predicted_category) %>%
  summarise(
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    se_log_engagement = sd(log_engagement, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 3) %>%  # Only show months with at least 3 posts
  mutate(
    frame_type = ifelse(is_group_binding_frame == 1, "Group-Binding Frame", "Other Frames")
  )

p2 <- ggplot(monthly_group_binding_stance, 
             aes(x = year_month_date, y = mean_log_engagement, 
                 color = interaction(predicted_category, frame_type), 
                 linetype = frame_type,
                 group = interaction(predicted_category, frame_type))) +
  geom_line(size = 1.2) +
  geom_point(size = 2, alpha = 0.7) +
  scale_color_manual(values = c(
    "Pro-Palestinian.Group-Binding Frame" = "darkblue",
    "Pro-Palestinian.Other Frames" = "lightblue",
    "Pro-Israeli.Group-Binding Frame" = "navy",
    "Pro-Israeli.Other Frames" = "skyblue",
    "Neutral.Group-Binding Frame" = "purple",
    "Neutral.Other Frames" = "plum"
  )) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  labs(
    title = "Evolution of Engagement by Group-Binding Framing and Stance Over Time",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Month",
    y = "Mean Log(Total Interactions)",
    color = "Stance × Frame",
    linetype = "Frame Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("engagement_evolution_group_binding_framing_stance.png", plot = p2, width = 16, height = 6, dpi = 300)
cat("Plot saved to: engagement_evolution_group_binding_framing_stance.png\n")

# Plot 3: Engagement by framing and stance over time
monthly_stance_framing <- data_clean %>%
  group_by(year_month_date, predicted_category, is_humanitarian_frame) %>%
  summarise(
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 5) %>%
  mutate(
    frame_type = ifelse(is_humanitarian_frame == 1, "Humanitarian", "Other")
  )

p3 <- ggplot(monthly_stance_framing, 
             aes(x = year_month_date, y = mean_log_engagement, 
                 color = interaction(predicted_category, frame_type), 
                 linetype = frame_type,
                 group = interaction(predicted_category, frame_type))) +
  geom_line(size = 1) +
  geom_point(size = 1.5, alpha = 0.7) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  scale_color_manual(values = c(
    "Pro-Palestinian.Humanitarian" = "darkred",
    "Pro-Palestinian.Other" = "lightcoral",
    "Pro-Israeli.Humanitarian" = "darkblue",
    "Pro-Israeli.Other" = "lightblue",
    "Neutral.Humanitarian" = "darkorange",
    "Neutral.Other" = "orange"
  )) +
  labs(
    title = "Evolution of Engagement by Framing and Stance Over Time",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Month",
    y = "Mean Log(Total Interactions)",
    color = "Stance × Frame",
    linetype = "Frame Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("engagement_evolution_by_stance_framing.png", plot = p3, width = 16, height = 6, dpi = 300)
cat("Plot saved to: engagement_evolution_by_stance_framing.png\n")

# Plot 4: Scatter plot with regression lines by framing and stance
p4 <- ggplot(data_clean, 
             aes(x = months_since_start, y = log_engagement, 
                 color = interaction(factor(is_humanitarian_frame, 
                                            labels = c("Other Frames", "Humanitarian Frame")),
                                     predicted_category))) +
  geom_point(alpha = 0.05, size = 0.3) +
  geom_smooth(method = "lm", se = TRUE, size = 1.2) +
  scale_color_manual(values = c(
    "Other Frames.Pro-Palestinian" = "lightcoral",
    "Humanitarian Frame.Pro-Palestinian" = "darkred",
    "Other Frames.Pro-Israeli" = "lightblue",
    "Humanitarian Frame.Pro-Israeli" = "darkblue",
    "Other Frames.Neutral" = "orange",
    "Humanitarian Frame.Neutral" = "darkorange"
  )) +
  labs(
    title = "Engagement vs Time by Framing Type and Stance",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Months Since Start",
    y = "Log(Total Interactions)",
    color = "Frame × Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    legend.position = "right"
  )

ggsave("engagement_scatter_by_framing_stance.png", plot = p4, width = 14, height = 6, dpi = 300)
cat("Plot saved to: engagement_scatter_by_framing_stance.png\n")

# Plot 5: Faceted plot by stance showing framing effects
p5 <- ggplot(data_clean, 
             aes(x = months_since_start, y = log_engagement, 
                 color = factor(is_humanitarian_frame, 
                               labels = c("Other Frames", "Humanitarian Frame")))) +
  geom_point(alpha = 0.05, size = 0.3) +
  geom_smooth(method = "lm", se = TRUE, size = 1.2) +
  facet_wrap(~ predicted_category, ncol = 3) +
  scale_color_manual(values = c("Other Frames" = "gray50", "Humanitarian Frame" = "darkred")) +
  labs(
    title = "Engagement vs Time by Framing Type (Faceted by Stance)",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Months Since Start",
    y = "Log(Total Interactions)",
    color = "Frame Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  )

ggsave("engagement_scatter_faceted_by_stance.png", plot = p5, width = 16, height = 5, dpi = 300)
cat("Plot saved to: engagement_scatter_faceted_by_stance.png\n")

# Plot 6: Stance evolution over time
p6 <- ggplot(monthly_stance, aes(x = year_month_date)) +
  geom_line(aes(y = pro_pal_prop, color = "Pro-Palestinian"), size = 1.2) +
  geom_line(aes(y = pro_isr_prop, color = "Pro-Israeli"), size = 1.2) +
  geom_line(aes(y = neutral_prop, color = "Neutral"), size = 1.2) +
  geom_point(aes(y = pro_pal_prop, color = "Pro-Palestinian"), size = 2, alpha = 0.7) +
  geom_point(aes(y = pro_isr_prop, color = "Pro-Israeli"), size = 2, alpha = 0.7) +
  geom_point(aes(y = neutral_prop, color = "Neutral"), size = 2, alpha = 0.7) +
  scale_color_manual(values = c("Pro-Palestinian" = "forestgreen", 
                               "Pro-Israeli" = "royalblue",
                               "Neutral" = "darkorange")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0, 1)) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  labs(
    title = "Evolution of Political Stance Over Time",
    subtitle = "Filtered: Accounts with followers (51+ posts) | Testing shift to Pro-Palestinian",
    x = "Month",
    y = "Proportion of Posts",
    color = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("stance_evolution_over_time.png", plot = p6, width = 14, height = 6, dpi = 300)
cat("Plot saved to: stance_evolution_over_time.png\n")

# Plot 7: Engagement by stance over time
monthly_engagement_stance <- data_clean %>%
  group_by(year_month_date, predicted_category) %>%
  summarise(
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    se_log_engagement = sd(log_engagement, na.rm = TRUE) / sqrt(n()),
    n = n(),
    .groups = "drop"
  ) %>%
  filter(n >= 5)

p7 <- ggplot(monthly_engagement_stance, 
             aes(x = year_month_date, y = mean_log_engagement, 
                 color = predicted_category, group = predicted_category)) +
  geom_line(size = 1.2) +
  geom_point(size = 2, alpha = 0.7) +
  geom_ribbon(aes(ymin = mean_log_engagement - se_log_engagement,
                  ymax = mean_log_engagement + se_log_engagement,
                  fill = predicted_category),
              alpha = 0.2, linetype = 0) +
  scale_color_manual(values = c("Pro-Palestinian" = "forestgreen", 
                               "Pro-Israeli" = "royalblue",
                               "Neutral" = "darkorange")) +
  scale_fill_manual(values = c("Pro-Palestinian" = "forestgreen", 
                               "Pro-Israeli" = "royalblue",
                               "Neutral" = "darkorange")) +
  scale_x_date(
    limits = x_axis_limits,
    date_labels = "%b %Y",
    date_breaks = "2 months"
  ) +
  labs(
    title = "Evolution of Engagement by Stance Over Time",
    subtitle = "Filtered: Accounts with followers (51+ posts)",
    x = "Month",
    y = "Mean Log(Total Interactions)",
    color = "Stance",
    fill = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggsave("engagement_evolution_by_stance.png", plot = p7, width = 14, height = 6, dpi = 300)
cat("Plot saved to: engagement_evolution_by_stance.png\n")

engagement_by_stance_current <- data_clean %>%
  group_by(predicted_category) %>%
  summarise(
    mean_engagement = mean(total_interactions, na.rm = TRUE),
    median_engagement = median(total_interactions, na.rm = TRUE),
    mean_log_engagement = mean(log_engagement, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

p8 <- ggplot(data_clean, 
             aes(x = factor(is_pro_palestinian, labels = c("Not Pro-Palestinian", "Pro-Palestinian")),
                 y = log_engagement,
                 fill = factor(is_pro_palestinian))) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 4, color = "red") +
  scale_fill_manual(values = c("0" = "gray70", "1" = "forestgreen")) +
  labs(
    title = "Engagement Levels by Pro-Palestinian Stance",
    subtitle = "Testing if higher engagement is associated with Pro-Palestinian stance",
    x = "Stance",
    y = "Log(Total Interactions)",
    fill = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    legend.position = "none"
  )

ggsave("engagement_by_pro_palestinian_stance.png", plot = p8, width = 10, height = 6, dpi = 300)
cat("Plot saved to: engagement_by_pro_palestinian_stance.png\n")

# Plot 9: Engagement leading to Humanitarian Framing
p9 <- ggplot(data_clean, 
             aes(x = factor(is_humanitarian_frame, labels = c("Other Frames", "Humanitarian Frame")),
                 y = log_engagement,
                 fill = factor(is_humanitarian_frame))) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  stat_summary(fun = mean, geom = "point", shape = 20, size = 4, color = "red") +
  scale_fill_manual(values = c("0" = "gray70", "1" = "darkred")) +
  labs(
    title = "Engagement Levels by Humanitarian Framing",
    subtitle = "Testing if higher engagement is associated with humanitarian framing",
    x = "Frame Type",
    y = "Log(Total Interactions)",
    fill = "Frame"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    legend.position = "none"
  )

ggsave("engagement_by_humanitarian_framing.png", plot = p9, width = 10, height = 6, dpi = 300)
cat("Plot saved to: engagement_by_humanitarian_framing.png\n")

if (sum(!is.na(data_clean$lag1_log_engagement)) > 100) {
  data_clean$lag_engagement_bin <- cut(data_clean$lag1_log_engagement,
                                       breaks = quantile(data_clean$lag1_log_engagement, 
                                                        probs = seq(0, 1, 0.2), na.rm = TRUE),
                                       labels = c("Very Low", "Low", "Medium", "High", "Very High"),
                                       include.lowest = TRUE)
  
  lag_engagement_effect <- data_clean %>%
    filter(!is.na(lag_engagement_bin)) %>%
    group_by(lag_engagement_bin) %>%
    summarise(
      pro_pal_prop = mean(is_pro_palestinian, na.rm = TRUE),
      humanitarian_prop = mean(is_humanitarian_frame, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    )
  
  p10_data <- lag_engagement_effect %>%
    pivot_longer(cols = c(pro_pal_prop, humanitarian_prop),
                names_to = "outcome",
                values_to = "proportion") %>%
    mutate(
      outcome_label = ifelse(outcome == "pro_pal_prop", "Pro-Palestinian Stance", "Humanitarian Framing")
    )
  
  p10 <- ggplot(p10_data, 
               aes(x = lag_engagement_bin, y = proportion, fill = outcome_label)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
    scale_fill_manual(values = c("Pro-Palestinian Stance" = "forestgreen", 
                                 "Humanitarian Framing" = "darkred")) +
    scale_y_continuous(labels = scales::percent_format()) +
    labs(
      title = "Effect of Previous Post Engagement on Current Stance and Framing",
      subtitle = "Does high engagement in previous post lead to Pro-Palestinian stance or humanitarian framing?",
      x = "Lagged Engagement Level (Previous Post)",
      y = "Proportion",
      fill = "Outcome"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "right"
    )
  
  ggsave("lagged_engagement_effect_on_stance_framing.png", plot = p10, width = 12, height = 6, dpi = 300)
  cat("Plot saved to: lagged_engagement_effect_on_stance_framing.png\n")
}

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n\n=== SUMMARY ===\n\n")
cat("Research Question: How has the relationship between framing and user engagement evolved over time?\n\n")

cat("Filters Applied:\n")
cat("  1. Excluded Personal Profile accounts (kept only accounts with followers)\n")
cat("  2. Kept only accounts with 51+ posts (Intensive Participation)\n")
cat("  Final dataset:", nrow(data_clean), "posts from", length(unique(data_clean[[facebook_id_col]])), "accounts\n\n")

cat("Key Findings:\n\n")

cat("RESEARCH QUESTION: Does Engagement Lead to Pro-Palestinian Stance and/or Humanitarian Framing?\n\n")

cat("DOES ENGAGEMENT LEAD TO PRO-PALESTINIAN STANCE?\n")
cat("Model A1 (Current Engagement):\n")
cat("  - Engagement coefficient:", round(coef(model_a1)[2], 4), 
    " (p =", round(summary_model_a1$coefficients[2,4], 6), ")\n")
cat("  - Odds Ratio:", round(odds_engagement_a1, 4), "\n")
if (summary_model_a1$coefficients[2,4] < 0.05) {
  cat("  → Higher engagement is associated with Pro-Palestinian stance\n")
} else {
  cat("  → No significant association between engagement and Pro-Palestinian stance\n")
}

if (length(coef(model_a2)) > 1) {
  cat("\nModel A2 (Lagged Engagement - Previous Post):\n")
  cat("  - Lagged engagement coefficient:", round(coef(model_a2)[2], 4), 
      " (p =", round(summary_model_a2$coefficients[2,4], 6), ")\n")
  if (summary_model_a2$coefficients[2,4] < 0.05) {
    cat("  → High engagement in previous post leads to Pro-Palestinian stance in current post\n")
  } else {
    cat("  → No evidence that previous engagement leads to Pro-Palestinian stance\n")
  }
}
cat("\n")

cat("DOES ENGAGEMENT LEAD TO HUMANITARIAN FRAMING?\n")
cat("Model B1 (Current Engagement):\n")
cat("  - Engagement coefficient:", round(coef(model_b1)[2], 4), 
    " (p =", round(summary_model_b1$coefficients[2,4], 6), ")\n")
cat("  - Odds Ratio:", round(odds_engagement_b1, 4), "\n")
if (summary_model_b1$coefficients[2,4] < 0.05) {
  cat("  → Higher engagement is associated with humanitarian framing\n")
} else {
  cat("  → No significant association between engagement and humanitarian framing\n")
}

if (length(coef(model_b2)) > 1) {
  cat("\nModel B2 (Lagged Engagement - Previous Post):\n")
  cat("  - Lagged engagement coefficient:", round(coef(model_b2)[2], 4), 
      " (p =", round(summary_model_b2$coefficients[2,4], 6), ")\n")
  if (summary_model_b2$coefficients[2,4] < 0.05) {
    cat("  → High engagement in previous post leads to humanitarian framing in current post\n")
  } else {
    cat("  → No evidence that previous engagement leads to humanitarian framing\n")
  }
}
cat("\n")

cat("STANCE EVOLUTION (Time Effect):\n")
cat("  - Time coefficient (Pro-Palestinian stance):", round(coef(model_stance)[2], 4), 
    " (p =", round(summary_model_stance$coefficients[2,4], 6), ")\n")
cat("  - Odds Ratio:", round(odds_ratio_stance, 4), "\n")
if (summary_model_stance$coefficients[2,4] < 0.05) {
  cat("  → Accounts significantly shift toward Pro-Palestinian stance over time\n")
  cat("  → Each month increases odds of Pro-Palestinian by", 
      round((odds_ratio_stance - 1) * 100, 2), "%\n", sep = "")
} else {
  cat("  → No significant shift toward Pro-Palestinian stance over time\n")
}
cat("\n")

cat("ENGAGEMENT AND FRAMING:\n")
cat("Model 1 (Humanitarian Framing with Stance):\n")
cat("  - Humanitarian framing effect:", round(coef(model1)[2], 4), 
    " (p =", round(summary_model1$coefficients[2,4], 6), ")\n")
cat("  - Time effect:", round(coef(model1)[3], 4), 
    " (p =", round(summary_model1$coefficients[3,4], 6), ")\n")
if (length(coef(model1)) > 4) {
  cat("  - Interaction (Framing × Time):", round(coef(model1)[4], 4), 
      " (p =", round(summary_model1$coefficients[4,4], 6), ")\n")
  if (summary_model1$coefficients[4,4] < 0.05) {
    cat("  → The relationship between humanitarian framing and engagement ", 
        ifelse(coef(model1)[4] > 0, "strengthens", "weakens"),
        " over time\n", sep = "")
  }
}
cat("\n")

cat("Model 2 (Group-Binding Framing with Stance):\n")
cat("  - Group-binding framing effect:", round(coef(model2)[2], 4), 
    " (p =", round(summary_model2$coefficients[2,4], 6), ")\n")
cat("  - Time effect:", round(coef(model2)[3], 4), 
    " (p =", round(summary_model2$coefficients[3,4], 6), ")\n")
if (length(coef(model2)) > 4) {
  cat("  - Interaction (Framing × Time):", round(coef(model2)[4], 4), 
      " (p =", round(summary_model2$coefficients[4,4], 6), ")\n")
  if (summary_model2$coefficients[4,4] < 0.05) {
    cat("  → The relationship between group-binding framing and engagement ", 
        ifelse(coef(model2)[4] > 0, "strengthens", "weakens"),
        " over time\n", sep = "")
  }
}

cat("\nAnalysis complete! Check generated plots and model summaries above.\n")

