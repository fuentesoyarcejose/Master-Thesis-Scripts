# R Script: Moral Usage by Account Reach Type
# This script analyzes and visualizes the use of different moral foundations by account reach type
# Account types are loaded from AccountsandMetadata.csv

# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(scales)

# Read the main dataset
data <- read_csv("/home/jose/Documents/UNI/Mémoire/Scripts/finaldata_with_sentiment.csv", 
                 locale = locale(encoding = "UTF-8"))

cat("Total number of rows in main dataset:", nrow(data), "\n")

# Read the account metadata file (semicolon-delimited)
account_metadata <- read_delim("/home/jose/Documents/UNI/Mémoire/Scripts/AccountsandMetadata.csv", 
                                delim = ";",
                                locale = locale(encoding = "UTF-8"))

cat("Total number of accounts in metadata:", nrow(account_metadata), "\n")

# Check Account Type distribution in metadata
cat("\nAccount Type distribution in metadata:\n")
print(table(account_metadata$`Account Type`, useNA = "ifany"))

if ("Facebook Id" %in% names(data)) {
  data$facebook_id <- as.numeric(data$`Facebook Id`)
} else {
  stop("Facebook Id column not found in main dataset")
}

account_metadata$facebook_id <- as.numeric(account_metadata$`Facebook Id`)

# Join datasets
data <- data %>%
  left_join(account_metadata %>% 
              select(facebook_id, `Account Type`), 
            by = "facebook_id")

cat("\nRows after joining with account metadata:", nrow(data), "\n")
cat("Rows with Account Type:", sum(!is.na(data$`Account Type`)), "\n")

# Check the predominant moral column (highest score)
cat("\nUnique predominant morals found:\n")
print(table(data$`highest score`, useNA = "ifany"))

# Function to map individual moral foundations
map_moral_to_category <- function(moral) {
  case_when(
    moral %in% c("Authority") ~ "Authority",
    moral %in% c("Loyalty") ~ "Loyalty",
    moral %in% c("Care") ~ "Care",
    moral %in% c("Harm") ~ "Harm",
    moral %in% c("Purity") ~ "Purity",
    TRUE ~ "Others"
  )
}

data_clean <- data %>%
  filter(!is.na(`Account Type`) & 
         `Account Type` != "" &
         !is.na(`highest score`) & 
         `highest score` != "") %>%
  mutate(
    account_reach_type = case_when(
      `Account Type` == "Global Reach/Global Broadcaster" ~ "Mass Reach Account",
      `Account Type` == "National Reach/Macro Influencer" ~ "High Reach Account",
      `Account Type` == "Regional Reach/Meso Influencer" ~ "Medium Reach Account",
      `Account Type` == "Local Community/Micro Influencer" ~ "Medium Reach Account",
      `Account Type` == "Low Reach Community/Emerging Voice" ~ "Low Reach Account",
      `Account Type` == "Personal Profile" ~ "Individual Account",
      TRUE ~ `Account Type`
    ),
    account_type = `Account Type`,
    moral_category = map_moral_to_category(`highest score`)
  )

cat("\nRows after cleaning:", nrow(data_clean), "\n")

# Check account reach type distribution
cat("\nAccount Reach Type distribution:\n")
account_type_counts <- table(data_clean$account_reach_type, useNA = "ifany")
print(account_type_counts)

reach_order <- c("Mass Reach Account", "High Reach Account", "Medium Reach Account", 
                 "Low Reach Account", "Individual Account")

moral_by_account <- data_clean %>%
  group_by(account_reach_type, moral_category) %>%
  summarise(count = n(), .groups = "drop") %>%
  group_by(account_reach_type) %>%
  mutate(
    total_posts = sum(count),
    percentage = (count / total_posts) * 100
  ) %>%
  ungroup() %>%
  mutate(account_reach_type = factor(account_reach_type, levels = reach_order)) %>%
  arrange(account_reach_type, desc(percentage))

min_posts <- 50
account_type_filtered <- moral_by_account %>%
  filter(total_posts >= min_posts) %>%
  distinct(account_reach_type) %>%
  pull(account_reach_type)

moral_by_account_filtered <- moral_by_account %>%
  filter(account_reach_type %in% account_type_filtered)

cat("\nAccount reach types with at least", min_posts, "posts:\n")
print(unique(moral_by_account_filtered$account_reach_type))

# Display summary table
cat("\nMoral Usage by Account Reach Type (%):\n")
print(moral_by_account_filtered %>%
  select(account_reach_type, moral_category, percentage, total_posts) %>%
  pivot_wider(names_from = moral_category, values_from = percentage, values_fill = 0) %>%
  arrange(account_reach_type))

# Color palette for morals
moral_colors <- c(
  "Authority" = "#8DD3C7",
  "Loyalty"   = "#FB8072",
  "Care"      = "#A6D854",
  "Harm"      = "#B3B3DE",
  "Purity"    = "#80B1D3",
  "Others"    = "#FDB462"
)

# Plot 1: Stacked Bar Chart - Percentage of each moral by account reach type
plot1 <- ggplot(moral_by_account_filtered, 
                aes(x = account_reach_type, 
                    y = percentage, 
                    fill = moral_category)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = moral_colors, name = "Moral Category") +
  scale_x_discrete(limits = reach_order) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title = "Moral Usage by Account Reach Type (% of Posts)",
    x = "Account Reach Type",
    y = "Percentage of Posts",
    fill = "Moral Category"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    legend.position = "right",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

print(plot1)
ggsave("/home/jose/Documents/UNI/Mémoire/Scripts/morals_by_account_type_stacked.png", 
       plot = plot1, width = 12, height = 8, dpi = 300)

# Plot 2: Grouped Bar Chart - Side-by-side comparison
plot2 <- ggplot(moral_by_account_filtered, 
                aes(x = account_reach_type, 
                    y = percentage, 
                    fill = moral_category)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = moral_colors, name = "Moral Category") +
  scale_x_discrete(limits = reach_order) +
  scale_y_continuous(labels = percent_format(scale = 1)) +
  labs(
    title = "Moral Usage by Account Reach Type - Side-by-Side Comparison",
    x = "Account Reach Type",
    y = "Percentage of Posts",
    fill = "Moral Category"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    legend.position = "right",
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

print(plot2)
ggsave("/home/jose/Documents/UNI/Mémoire/Scripts/morals_by_account_type_grouped.png", 
       plot = plot2, width = 14, height = 8, dpi = 300)

# Plot 3: Heatmap - Visual representation of moral usage intensity
plot3 <- ggplot(moral_by_account_filtered, 
                aes(x = moral_category, 
                    y = account_reach_type, 
                    fill = percentage)) +
  geom_tile(color = "white", linewidth = 0.5) +
  scale_fill_gradient2(low = "white", 
                       mid = "lightblue", 
                       high = "darkblue",
                       midpoint = median(moral_by_account_filtered$percentage),
                       name = "% of Posts") +
  scale_y_discrete(limits = rev(reach_order)) +
  labs(
    title = "Moral Usage Heatmap by Account Reach Type",
    x = "Moral Category",
    y = "Account Reach Type",
    fill = "% of Posts"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    panel.grid = element_blank()
  )

print(plot3)
ggsave("/home/jose/Documents/UNI/Mémoire/Scripts/morals_by_account_type_heatmap.png", 
       plot = plot3, width = 10, height = 8, dpi = 300)

account_summary <- data_clean %>%
  filter(account_reach_type %in% account_type_filtered) %>%
  group_by(account_reach_type) %>%
  summarise(total_posts = n(), .groups = "drop") %>%
  mutate(account_reach_type = factor(account_reach_type, levels = reach_order)) %>%
  arrange(account_reach_type)

plot4 <- ggplot(account_summary, 
                aes(x = account_reach_type, 
                    y = total_posts)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  scale_x_discrete(limits = reach_order) +
  scale_y_continuous(labels = number_format(big.mark = ".", decimal.mark = ",")) +
  labs(
    title = "Total Posts by Account Reach Type",
    x = "Account Reach Type",
    y = "Number of Posts"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank()
  )

print(plot4)
ggsave("/home/jose/Documents/UNI/Mémoire/Scripts/posts_by_account_type.png", 
       plot = plot4, width = 10, height = 8, dpi = 300)

cat("\n\nAnalysis complete! Generated 4 visualizations:\n")
cat("1. Stacked bar chart: morals_by_account_type_stacked.png\n")
cat("2. Grouped bar chart: morals_by_account_type_grouped.png\n")
cat("3. Heatmap: morals_by_account_type_heatmap.png\n")
cat("4. Post counts: posts_by_account_type.png\n")

