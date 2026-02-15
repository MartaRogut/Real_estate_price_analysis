import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def load_data():
    files = glob.glob("data/*.csv")

    dfs = []

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    print(full_df.shape)
    print(full_df.info())
    print(full_df.head())
    print(full_df.describe())
    # zapis próbki danych do repo
    full_df.sample(1000).to_csv("data/sample_data.csv", index=False)
    return full_df


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def clean_data(full_df):
    print("Shape before cleaning:", full_df.shape)
    duplicates = full_df.duplicated().sum()
    print("Duplicates:", duplicates)

    full_df = full_df.drop_duplicates()
    # usuwam braki w kolumnach kluczowych
    full_df = full_df.dropna(subset=["price", "squareMeters", "rooms"])
    # usunięcie nielicznych/ błędnych infromacji
    full_df = full_df[
        (full_df["price"] > 10000) &          # usuwam podejrzanie tanie mieszkania
        (full_df["squareMeters"] > 15) &      # za małe metraże
        (full_df["squareMeters"] < 300) &     # ogromne mieszkania
        (full_df["rooms"] > 0) &
        (full_df["rooms"] < 10)
    ]
    full_df["price_per_m2"] = full_df["price"] / full_df["squareMeters"]  # cena za m^2

    # usuwam outliery
    full_df = remove_outliers_iqr(full_df, "price")
    full_df = remove_outliers_iqr(full_df, "price_per_m2")

    # reset index
    full_df = full_df.reset_index(drop=True)

    # sprawdzenie
    print("Shape after cleaning:", full_df.shape)
    print(full_df.describe())

    # Usunięcie absurdalnych price_per_m2
    full_df = full_df[
        (full_df["price_per_m2"] > 1000) &
        (full_df["price_per_m2"] < 40000)
    ]

    full_df = full_df.reset_index(drop=True)

    print("Final shape:", full_df.shape)
    print(full_df.describe())

    return full_df


def run_eda(full_df):
    #                               Od czego zależy cena mieszkań w Polsce
    # porównanie cena vs metraż

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="squareMeters", y="price", data=full_df, alpha=0.3)

    plt.title("Apartment Price vs Size")
    plt.xlabel("Square meters")
    plt.ylabel("Price")
    plt.show()

    # linia trendu
    plt.figure(figsize=(8, 6))
    sns.regplot(x="squareMeters", y="price", data=full_df, scatter_kws={"alpha": 0.2})

    plt.title("Price vs Size with Trend Line")
    plt.xlabel("Square meters")
    plt.ylabel("Price")
    plt.savefig("price_vs_size.png", bbox_inches="tight")
    plt.show()

    


    # korelacja
    correlation = full_df["squareMeters"].corr(full_df["price"])
    print("Correlation between size and price:", correlation)

    # cena za m^2 vs metraż
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="squareMeters", y="price_per_m2", data=full_df, alpha=0.3)

    plt.title("Price per m2 vs Size")
    plt.xlabel("Square meters")
    plt.ylabel("Price per m2")
    plt.show()

    #                                           cena vs liczba pokoi

    # wykres rozrzutu
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="rooms", y="price", data=full_df, alpha=0.3)

    plt.title("Apartment Price vs Number of Rooms")
    plt.xlabel("Number of rooms")
    plt.ylabel("Price")
    plt.show()

    # średnia cena dla liczby pokoi
    full_df.groupby("rooms")["price"].mean()
    # wykres słupkowy
    avg_price_rooms = full_df.groupby("rooms")["price"].mean()

    plt.figure(figsize=(8, 6))
    avg_price_rooms.plot(kind="bar")

    plt.title("Average Price by Number of Rooms")
    plt.xlabel("Number of rooms")
    plt.ylabel("Average price")
    plt.show()

    # korelacja
    corr_rooms = full_df["rooms"].corr(full_df["price"])
    print("Correlation between rooms and price:", corr_rooms)

    #                                                           cena vs odległość od centrum
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="centreDistance", y="price", data=full_df, alpha=0.3)

    plt.title("Apartment Price vs Distance from City Centre")
    plt.xlabel("Distance from centre")
    plt.ylabel("Price")
    plt.show()

    # linia trendu
    plt.figure(figsize=(8, 6))
    sns.regplot(x="centreDistance", y="price", data=full_df, scatter_kws={"alpha": 0.2})

    plt.title("Price vs Distance from Centre (Trend Line)")
    plt.xlabel("Distance from centre")
    plt.ylabel("Price")
    plt.savefig("price_vs_distance.png", dpi=300)
    plt.show()

    #korelacja
    corr_centre = full_df["centreDistance"].corr(full_df["price"])
    print("Correlation between distance and price:", corr_centre)

    #                                                           Wpływ udogodnień tj. balkon, winda, parking, ochrona

    # średnia cena w zależności od udogodnienia
    amenities = ["hasElevator", "hasSecurity", "hasParkingSpace"]

price_premium = {}

for col in amenities:
    mean_prices = full_df.groupby(col)["price"].mean()
    premium = mean_prices["yes"] - mean_prices["no"]
    price_premium[col] = premium

plt.figure(figsize=(8,6))
plt.bar(price_premium.keys(), price_premium.values())

plt.title("Price Premium Generated by Amenities")
plt.ylabel("Average Price Difference (YES vs NO)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("amenities_price_premium.png", dpi=300)
plt.show()

    #                                           MACIERZ KORELACJI

    plt.figure(figsize=(12, 8))
    sns.heatmap(full_df.corr(numeric_only=True),
                cmap="coolwarm",
                annot=False)

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig("correlation_matrix.png", dpi=300)
    plt.show()

    # korelacja z ceną
    corr_with_price = full_df.corr(numeric_only=True)["price"].sort_values(ascending=False)
    print(corr_with_price)


def train_model(full_df):
    #                                               MODEL PREDYKCYJNY
    # wybieram zmienne i zamieniam yes/no na 0/1
    features = [
        "squareMeters",
        "rooms",
        "centreDistance",
        "floor",
        "buildYear",
        "hasElevator",
        "hasParkingSpace",
        "hasSecurity"
    ]
    full_df_model = full_df.copy()

    binary_cols = ["hasElevator", "hasParkingSpace", "hasSecurity"]

    for col in binary_cols:
        full_df_model[col] = full_df_model[col].map({"yes": 1, "no": 0})

    # przygotowuje dane
    X = full_df_model[features]
    y = full_df_model["price"]

    # usuwamy ewentualne NaN
    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # regresja liniowa
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("R2 score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    # wspołczynniki modelu
    coefficients = pd.Series(model.coef_, index=features)
    print(coefficients.sort_values(ascending=False))

    plt.figure(figsize=(8,6))
    coefficients.sort_values().plot(kind="barh")
    plt.title("Feature Impact on Apartment Price (Linear Regression)")
    plt.xlabel("Coefficient value")
    plt.tight_layout()
    plt.savefig("model_feature_impact.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.show()


    return coefficients, y_test, y_pred


def summarize_results(coefficients, y_test, y_pred):
    #                                               PODSUMOWANIE
    # Wykres słupkowy porównujący najważniejsze czynniki
    coefficients.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.title("Feature Impact on Apartment Price (Linear Regression)")
    plt.xlabel("Coefficient value")
    plt.show()

    # Wykres rzeczywiste vs przewidywane cehcy
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.show()


def main():
    full_df = load_data()
    full_df = clean_data(full_df)
    run_eda(full_df)
    coefficients, y_test, y_pred = train_model(full_df)
    summarize_results(coefficients, y_test, y_pred)


if __name__ == "__main__":
    main()






