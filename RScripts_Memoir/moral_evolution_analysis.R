# R Script: Evolution of Highest Score Morals per Year-Month
# This script analyzes the percentage of predominant moral foundations in posts per year-month

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(lubridate)

# Read the CSV file
data <- read_csv("/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv", 
                 locale = locale(encoding = "UTF-8"))

# Display basic information about the data
cat("Total number of rows:", nrow(data), "\n")
cat("Date range:", min(data$`Post Created Date`, na.rm = TRUE), "to", 
    max(data$`Post Created Date`, na.rm = TRUE), "\n")

# Extract year-month from Post Created Date
data$date <- as.Date(data$`Post Created Date`, format = "%Y-%m-%d")
data$year_month <- format(data$date, "%Y-%m")
data$year_month_date <- as.Date(paste0(data$year_month, "-01"))

# Check the predominant moral column (highest score)
cat("\nUnique predominant morals found:\n")
print(table(data$`highest score`, useNA = "ifany"))

# Function to map individual moral foundations - keep main ones, group others
map_moral_to_category <- function(moral) {
  case_when(
    moral %in% c("Authority") ~ "Authority",
    moral %in% c("Loyalty") ~ "Loyalty",
    moral %in% c("Care") ~ "Care",
    moral %in% c("Harm") ~ "Harm",
    moral %in% c("Purity") ~ "Purity",
    TRUE ~ "Others"  # Group all others (Betrayal, Subversion, Degradation, Cheating, Fairness, etc.)
  )
}

# Filter out rows with missing year-month or predominant moral
data_clean <- data %>%
  filter(!is.na(year_month) & !is.na(`highest score`) & `highest score` != "") %>%
  mutate(moral_category = map_moral_to_category(`highest score`))

cat("\nRows after cleaning:", nrow(data_clean), "\n")
cat("\nMoral categories distribution:\n")
print(table(data_clean$moral_category, useNA = "ifany"))

# Calculate the percentage of each moral category per year-month
moral_evolution <- data_clean %>%
  group_by(year_month, year_month_date, moral_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(year_month, year_month_date) %>%
  mutate(
    total_posts = sum(count),
    percentage = (count / total_posts) * 100
  ) %>%
  ungroup() %>%
  arrange(year_month_date)

# Get total posts per month for bar chart
posts_per_month <- moral_evolution %>%
  group_by(year_month, year_month_date) %>%
  summarise(total_posts = first(total_posts), .groups = "drop") %>%
  arrange(year_month_date)

# Calculate scaling factor for dual axis (scale post counts to percentage range)
max_posts <- max(posts_per_month$total_posts)
scale_factor <- 100 / max_posts

# Custom number formatter with dots as thousand separators (European format)
format_number_dots <- function(x) {
  format(x, big.mark = ".", decimal.mark = ",", scientific = FALSE)
}

# Get all unique year-month dates for x-axis breaks
all_months <- unique(moral_evolution$year_month_date)
all_months <- sort(all_months)

# Display summary statistics
cat("\nSummary by year-month:\n")
print(moral_evolution %>%
  group_by(year_month) %>%
  summarise(
    total_posts = first(total_posts),
    unique_morals = n(),
    .groups = "drop"
  ))

# Create a wide format for better visualization
moral_evolution_wide <- moral_evolution %>%
  select(year_month, moral_category, percentage) %>%
  pivot_wider(names_from = moral_category, values_from = percentage, values_fill = 0) %>%
  arrange(year_month)

# Display the results table
cat("\nEvolution of Highest Score Morals (% per year-month):\n")
print(moral_evolution_wide)

# Save results to CSV
write_csv(moral_evolution_wide, "moral_evolution_per_year_month.csv")
cat("\nResults saved to: moral_evolution_per_year_month.csv\n")

# Define color palette
muted_colors <- c(
  "Authority" = "#8DD3C7",
  "Loyalty"   = "#FB8072",
  "Care"      = "#A6D854",
  "Harm"      = "#B3B3DE",
  "Purity"    = "#80B1D3",
  "Others"    = "#FDB462"
)

# Stance colors (for potential future use)
stance_colors <- c(
  "Pro-Palestinian" = "forestgreen",
  "Pro-Israeli" = "royalblue",
  "Neutral" = "darkorange"
)

# Create visualization: Line plot showing evolution with post count bars
p1 <- ggplot() +
  # Add post count bars (scaled to percentage range)
  geom_bar(data = posts_per_month, 
           aes(x = year_month_date, y = total_posts * scale_factor), 
           stat = "identity", 
           fill = "gray90", 
           alpha = 0.3,
           width = 20) +
  # Add percentage lines
  geom_line(data = moral_evolution, 
            aes(x = year_month_date, y = percentage, color = moral_category, group = moral_category), 
            size = 1.2) +
  geom_point(data = moral_evolution, 
             aes(x = year_month_date, y = percentage, color = moral_category), 
             size = 2) +
  labs(
    title = "Evolution of Highest Score Morals (% of Total Post per Year-Month)",
    x = "Year-Month",
    y = "Percentage (%)",
    color = "Highest Moral Score in Post"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y.right = element_text(color = "gray50"),
    axis.title.y.right = element_text(color = "gray50"),
    legend.position = "right"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(scale = 1),
    sec.axis = sec_axis(~ . / scale_factor, name = "Number of Posts", labels = format_number_dots)
  ) +
  scale_x_date(date_labels = "%Y-%m", breaks = all_months) +
  scale_color_manual(values = muted_colors)

# Save the plot
ggsave("moral_evolution_line_plot.png", plot = p1, width = 14, height = 6, dpi = 300)
cat("Line plot saved to: moral_evolution_line_plot.png\n")

# Create visualization: Stacked area chart with post count bars
p2 <- ggplot() +
  # Add post count bars (scaled to percentage range)
  geom_bar(data = posts_per_month, 
           aes(x = year_month_date, y = total_posts * scale_factor), 
           stat = "identity", 
           fill = "gray90", 
           alpha = 0.3,
           width = 20) +
  # Add stacked area
  geom_area(data = moral_evolution, 
            aes(x = year_month_date, y = percentage, fill = moral_category), 
            alpha = 0.8, 
            position = "stack") +
  labs(
    title = "Evolution of Highest Score Morals (% of Total Post per Year-Month)",
    x = "Year-Month",
    y = "Percentage (%)",
    fill = "Highest Moral Score in Post"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y.right = element_text(color = "gray50"),
    axis.title.y.right = element_text(color = "gray50"),
    legend.position = "right"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(scale = 1),
    sec.axis = sec_axis(~ . / scale_factor, name = "Number of Posts", labels = format_number_dots)
  ) +
  scale_x_date(date_labels = "%Y-%m", breaks = all_months) +
  scale_fill_manual(values = muted_colors)

ggsave("moral_evolution_stacked_area.png", plot = p2, width = 14, height = 6, dpi = 300)
cat("Stacked area chart saved to: moral_evolution_stacked_area.png\n")

# Create visualization: Bar chart (grouped by year-month) with post count bars
# Get all year-months in order for the bar chart
all_year_months <- unique(moral_evolution$year_month)
all_year_months <- sort(all_year_months)

# Create a combined plot with post counts
posts_per_month_for_bars <- posts_per_month %>%
  mutate(year_month = format(year_month_date, "%Y-%m"))

p3 <- ggplot() +
  # Add post count bars (scaled to percentage range)
  geom_bar(data = posts_per_month_for_bars,
           aes(x = factor(year_month, levels = all_year_months), 
               y = total_posts * scale_factor), 
           stat = "identity", 
           fill = "gray90", 
           alpha = 0.3,
           width = 0.7) +
  # Add stacked percentage bars
  geom_bar(data = moral_evolution, 
           aes(x = factor(year_month, levels = all_year_months), 
               y = percentage, 
               fill = moral_category), 
           stat = "identity", 
           position = "stack",
           width = 0.7) +
  labs(
    title = "Evolution of Highest Score Morals (% of Total Post per Year-Month)",
    x = "Year-Month",
    y = "Percentage (%)",
    fill = "Highest Moral Score in Post"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y.right = element_text(color = "gray50"),
    axis.title.y.right = element_text(color = "gray50"),
    legend.position = "right"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(scale = 1),
    sec.axis = sec_axis(~ . / scale_factor, name = "Number of Posts", labels = format_number_dots)
  ) +
  scale_fill_manual(values = muted_colors)

# Also create a separate plot showing absolute post counts
p4 <- ggplot(posts_per_month, aes(x = year_month_date, y = total_posts)) +
  geom_bar(stat = "identity", fill = "#6B8E9F", alpha = 0.7) +
  # Add trend line (in red)
  geom_smooth(method = "loess", 
              se = FALSE, 
              color = "red", 
              size = 1.2,
              linetype = "dashed") +
  labs(
    title = "Number of Posts per Month",
    x = "Year-Month",
    y = "Number of Posts"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8)
  ) +
  scale_x_date(date_labels = "%Y-%m", breaks = all_months) +
  scale_y_continuous(labels = format_number_dots)

ggsave("moral_evolution_stacked_bar.png", plot = p3, width = 14, height = 6, dpi = 300)
cat("Stacked bar chart saved to: moral_evolution_stacked_bar.png\n")

ggsave("post_counts_per_month.png", plot = p4, width = 14, height = 6, dpi = 300)
cat("Post counts chart saved to: post_counts_per_month.png\n")

# Political Stance Analysis - Percentage per month
cat("\n=== Political Stance Analysis ===\n")

# Check predicted_category values
cat("\nUnique predicted_category values:\n")
print(table(data$predicted_category, useNA = "ifany"))

# Filter and clean stance data - use predicted_category directly
data_stance <- data %>%
  filter(!is.na(year_month) & !is.na(predicted_category) & predicted_category != "")

cat("\nStance distribution after cleaning:\n")
print(table(data_stance$predicted_category, useNA = "ifany"))

# Calculate percentage of each stance per year-month
stance_evolution <- data_stance %>%
  group_by(year_month, year_month_date, predicted_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(year_month, year_month_date) %>%
  mutate(
    total_posts = sum(count),
    percentage = (count / total_posts) * 100
  ) %>%
  ungroup() %>%
  arrange(year_month_date, predicted_category)

# Get all months for x-axis
all_months_stance <- unique(stance_evolution$year_month_date)
all_months_stance <- sort(all_months_stance)

# Create line chart for political stance percentages
p5 <- ggplot(stance_evolution, 
             aes(x = year_month_date, y = percentage, color = predicted_category, group = predicted_category)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  labs(
    title = "Evolution of Political Stance in Facebook Posts",
    subtitle = "% of Political Stance of Total Posts per Month",
    x = "Month",
    y = "Percentage (%)",
    color = "Stance"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, color = "gray50"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "right"
  ) +
  scale_y_continuous(labels = scales::percent_format(scale = 1)) +
  scale_x_date(date_labels = "%b %Y", breaks = all_months_stance) +
  scale_color_manual(values = stance_colors)

ggsave("political_stance_evolution_percentage.png", plot = p5, width = 14, height = 6, dpi = 300)
cat("Political stance percentage chart saved to: political_stance_evolution_percentage.png\n")

# Display detailed table with counts and percentages
cat("\nPolitical stance breakdown (Count and Percentage per year-month):\n")
print(stance_evolution %>%
  arrange(year_month_date, desc(percentage)) %>%
  select(year_month, predicted_category, count, total_posts, percentage) %>%
  mutate(percentage = round(percentage, 2)))

# Display detailed table with counts and percentages
cat("\nDetailed breakdown (Count and Percentage per year-month):\n")
print(moral_evolution %>%
  arrange(year_month_date, desc(percentage)) %>%
  select(year_month, moral_category, count, total_posts, percentage) %>%
  mutate(percentage = round(percentage, 2)))

# Moral Frames Analysis for Pro-Palestinian Posts
cat("\n=== Moral Frames Analysis for Pro-Palestinian Posts ===\n")

# Filter for Pro-Palestinian posts only
data_pro_pal <- data %>%
  filter(!is.na(year_month) & 
         !is.na(predicted_category) & 
         predicted_category != "" &
         predicted_category == "Pro-Palestinian" &
         !is.na(`highest score`) & 
         `highest score` != "") %>%
  mutate(moral_category = map_moral_to_category(`highest score`))

cat("\nPro-Palestinian posts after cleaning:", nrow(data_pro_pal), "\n")
cat("\nMoral categories distribution for Pro-Palestinian posts:\n")
print(table(data_pro_pal$moral_category, useNA = "ifany"))

# Calculate proportion of each moral frame per year-month for Pro-Palestinian posts
moral_pro_pal <- data_pro_pal %>%
  group_by(year_month, year_month_date, moral_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(year_month, year_month_date) %>%
  mutate(
    total_posts = sum(count),
    proportion = (count / total_posts) * 100
  ) %>%
  ungroup() %>%
  arrange(year_month_date, moral_category)

# Get total Pro-Palestinian posts per month for secondary axis
posts_pro_pal_per_month <- moral_pro_pal %>%
  group_by(year_month, year_month_date) %>%
  summarise(total_posts = first(total_posts), .groups = "drop") %>%
  arrange(year_month_date)

# Calculate scaling factor for dual axis
max_proportion <- max(moral_pro_pal$proportion, na.rm = TRUE)
max_posts_pro_pal <- max(posts_pro_pal_per_month$total_posts, na.rm = TRUE)
scale_factor_pro_pal <- max_proportion / max_posts_pro_pal

# Get all months for x-axis
all_months_pro_pal <- unique(moral_pro_pal$year_month_date)
all_months_pro_pal <- sort(all_months_pro_pal)

# Create line chart with secondary Y-axis for number of posts
p6 <- ggplot() +
  # Add post count bars (scaled to proportion range) for secondary axis
  geom_bar(data = posts_pro_pal_per_month, 
           aes(x = year_month_date, y = total_posts * scale_factor_pro_pal), 
           stat = "identity", 
           fill = "gray90", 
           alpha = 0.3,
           width = 20) +
  # Add proportion lines for each moral frame
  geom_line(data = moral_pro_pal, 
            aes(x = year_month_date, y = proportion, color = moral_category, group = moral_category), 
            size = 1.2) +
  geom_point(data = moral_pro_pal, 
             aes(x = year_month_date, y = proportion, color = moral_category), 
             size = 2) +
  labs(
    title = "Evolution of Predominant Moral Frames Over Time - Pro-Palestinian",
    x = "Month",
    y = "Proportion of Posts (%)",
    color = "Moral Frame"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y.right = element_text(color = "gray50"),
    axis.title.y.right = element_text(color = "gray50"),
    legend.position = "right"
  ) +
  scale_y_continuous(
    name = "Proportion of Posts (%)",
    labels = scales::percent_format(scale = 1),
    sec.axis = sec_axis(~ . / scale_factor_pro_pal, 
                       name = "Number of Posts", 
                       labels = format_number_dots)
  ) +
  scale_x_date(date_labels = "%b %Y", breaks = all_months_pro_pal) +
  scale_color_manual(values = muted_colors)

ggsave("moral_frames_pro_palestinian.png", plot = p6, width = 14, height = 6, dpi = 300)
cat("Moral frames for Pro-Palestinian posts chart saved to: moral_frames_pro_palestinian.png\n")

# Display detailed table
cat("\nMoral frames breakdown for Pro-Palestinian posts (Count and Proportion per year-month):\n")
print(moral_pro_pal %>%
  arrange(year_month_date, desc(proportion)) %>%
  select(year_month, moral_category, count, total_posts, proportion) %>%
  mutate(proportion = round(proportion, 2)))

# Moral Frames Analysis for Pro-Israeli Posts
cat("\n=== Moral Frames Analysis for Pro-Israeli Posts ===\n")

# Filter for Pro-Israeli posts only
data_pro_isr <- data %>%
  filter(!is.na(year_month) & 
         !is.na(predicted_category) & 
         predicted_category != "" &
         predicted_category == "Pro-Israeli" &
         !is.na(`highest score`) & 
         `highest score` != "") %>%
  mutate(moral_category = map_moral_to_category(`highest score`))

cat("\nPro-Israeli posts after cleaning:", nrow(data_pro_isr), "\n")
cat("\nMoral categories distribution for Pro-Israeli posts:\n")
print(table(data_pro_isr$moral_category, useNA = "ifany"))

# Calculate proportion of each moral frame per year-month for Pro-Israeli posts
moral_pro_isr <- data_pro_isr %>%
  group_by(year_month, year_month_date, moral_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(year_month, year_month_date) %>%
  mutate(
    total_posts = sum(count),
    proportion = (count / total_posts) * 100
  ) %>%
  ungroup() %>%
  arrange(year_month_date, moral_category)

# Get total Pro-Israeli posts per month for secondary axis
posts_pro_isr_per_month <- moral_pro_isr %>%
  group_by(year_month, year_month_date) %>%
  summarise(total_posts = first(total_posts), .groups = "drop") %>%
  arrange(year_month_date)

# Calculate scaling factor for dual axis
max_proportion_isr <- max(moral_pro_isr$proportion, na.rm = TRUE)
max_posts_pro_isr <- max(posts_pro_isr_per_month$total_posts, na.rm = TRUE)
scale_factor_pro_isr <- max_proportion_isr / max_posts_pro_isr

# Get all months for x-axis
all_months_pro_isr <- unique(moral_pro_isr$year_month_date)
all_months_pro_isr <- sort(all_months_pro_isr)

# Create line chart with secondary Y-axis for number of posts
p7 <- ggplot() +
  # Add post count bars (scaled to proportion range) for secondary axis
  geom_bar(data = posts_pro_isr_per_month, 
           aes(x = year_month_date, y = total_posts * scale_factor_pro_isr), 
           stat = "identity", 
           fill = "gray90", 
           alpha = 0.3,
           width = 20) +
  # Add proportion lines for each moral frame
  geom_line(data = moral_pro_isr, 
            aes(x = year_month_date, y = proportion, color = moral_category, group = moral_category), 
            size = 1.2) +
  geom_point(data = moral_pro_isr, 
             aes(x = year_month_date, y = proportion, color = moral_category), 
             size = 2) +
  labs(
    title = "Evolution of Predominant Moral Frames Over Time - Pro-Israeli",
    x = "Month",
    y = "Proportion of Posts (%)",
    color = "Moral Frame"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y.right = element_text(color = "gray50"),
    axis.title.y.right = element_text(color = "gray50"),
    legend.position = "right"
  ) +
  scale_y_continuous(
    name = "Proportion of Posts (%)",
    labels = scales::percent_format(scale = 1),
    sec.axis = sec_axis(~ . / scale_factor_pro_isr, 
                       name = "Number of Posts", 
                       labels = format_number_dots)
  ) +
  scale_x_date(date_labels = "%b %Y", breaks = all_months_pro_isr) +
  scale_color_manual(values = muted_colors)

ggsave("moral_frames_pro_israeli.png", plot = p7, width = 14, height = 6, dpi = 300)
cat("Moral frames for Pro-Israeli posts chart saved to: moral_frames_pro_israeli.png\n")

# Display detailed table
cat("\nMoral frames breakdown for Pro-Israeli posts (Count and Proportion per year-month):\n")
print(moral_pro_isr %>%
  arrange(year_month_date, desc(proportion)) %>%
  select(year_month, moral_category, count, total_posts, proportion) %>%
  mutate(proportion = round(proportion, 2)))

cat("\nAnalysis complete!\n")

