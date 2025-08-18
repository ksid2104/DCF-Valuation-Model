# =========================
# Imports
# =========================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web   # Données macro-économiques (PIB)

# Options pandas pour affichage complet
pd.set_option("display.max_rows", None)
pd.set_option('display.max_columns', None)

# =========================
# Variables initiales à définir par l'utilisateur
# =========================
ticker = input("Entrez le ticker de l'entreprise à analyser : ")  # Ticker de l'entreprise à analyser

# Prévision : nombre d'années à projeter pour les FCF
forecast_years = pd.date_range(
    start=pd.to_datetime(yf.Ticker("MSFT").financials.columns[0]),
    periods=int(input("Entrez le nombre d'années de prévision : ")),
    freq="Y",
    normalize=True
)[1:]

# Equity Risk Premium implicite (source Damodaran NYU Stern)
implied_ERP = float(input(
    "Entrez la valeur de l'ERP implicite (source : ➝ https://pages.stern.nyu.edu/~adamodar/) : "
))

# =========================
# Récupération des données financières historiques via yFinance
# =========================
income_statement = yf.Ticker(ticker).financials
balance_sheet = yf.Ticker(ticker).balance_sheet
cashflow = yf.Ticker(ticker).cashflow

# =========================
# Fonctions principales
# =========================

def FCF(income_statement=income_statement, cashflow=cashflow):
    """
    Calcul du Free Cash Flow (historique)
    FCF = EBIT*(1-t) + Amortissement - CAPEX + ΔBFR
    Note : ΔBFR déjà signé correctement dans yFinance
    """
    ebit = income_statement.loc["EBIT"]
    t = income_statement.loc["Tax Rate For Calcs"]
    amortization = cashflow.loc["Depreciation And Amortization"]
    capex = abs(cashflow.loc["Capital Expenditure"])
    delta_BFR = cashflow.loc["Change In Working Capital"]
    return (ebit*(1-t) + amortization - capex + delta_BFR).dropna()


def projected_fcf(fcf=FCF(), years=forecast_years):
    """
    Projection des FCF sur la période définie (ex: 5 ans)
    Basé sur le CAGR historique des FCF
    """
    start_fcf = fcf.iloc[-1]
    cagr = (fcf.iloc[-1] / fcf.iloc[0]) ** (1 / (len(fcf) - 1)) - 1
    values = [start_fcf*(1+cagr)**x for x in range(len(forecast_years))]
    years_dt = pd.to_datetime(years)
    return pd.Series(values, index=years_dt).sort_index(ascending=True)


def WACC(income_statement=income_statement, balance_sheet=balance_sheet, implied_ERP=implied_ERP):
    """
    Calcul du WACC (Weighted Average Cost of Capital)
    WACC = (E/(D+E))*Re + (D/(D+E))*Rd*(1-T)
    - Re calculé via CAPM avec ERP implicite
    - Rd = coût moyen de la dette
    """
    d = balance_sheet.loc["Total Debt"].iloc[0]
    e = yf.Ticker(ticker).info["marketCap"]
    rd = income_statement.loc["Interest Expense"].dropna().mean() / d

    # CAPM pour Re
    risk_free_rate = float(yf.download("^TNX", auto_adjust=True)["Close"].iloc[-1]/100)
    beta = yf.Ticker(ticker).info["beta"]
    rm = implied_ERP
    re = risk_free_rate + beta*rm
    t = income_statement.loc["Tax Rate For Calcs"].iloc[0]

    return float((e/(d+e))*re + (d/(d+e))*rd*(1-t))


def DCF(projected_fcf=projected_fcf(), WACC=WACC()):
    """
    Calcul des DCF pour la période projetée
    Actualisation des FCF avec le WACC
    """
    dcf = projected_fcf / (1 + WACC)**(np.arange(1, len(projected_fcf)+1))
    return dcf


def Terminal_Value(fcf=FCF(), projected_fcf=projected_fcf(), wacc=WACC()):
    """
    Calcul de la valeur terminale (TV)
    TV = FCF dernier an * (1+g) / (WACC - g)
    g = moyenne taux croissance PIB US + taux sans risque
    """
    gdp_values = pd.DataFrame(web.DataReader("GDPCA", "fred", start='1928-01-01'))
    gdp_growth = (gdp_values.iloc[-1]/gdp_values.iloc[0])**(1/(len(gdp_values)))-1
    risk_free_rate = float(yf.download("^TNX", auto_adjust=True)["Close"].iloc[-1]/100)
    g = (gdp_growth + risk_free_rate)/2
    tv = projected_fcf.iloc[-1] * (1 + g) / (wacc - g)
    return float(tv), float(g)


def Final_Valuation(dcf=DCF(), tv=Terminal_Value()[0], WACC=WACC()):
    """
    Valorisation finale = somme DCF + valeur terminale actualisée
    """
    discounted_tv = tv / (1 + WACC)**len(dcf)
    valuation = sum(dcf) + discounted_tv
    return valuation


def Equity_Value(valuation=Final_Valuation()):
    """
    Valeur des capitaux propres = Valorisation - Dette nette
    """
    net_debt = balance_sheet.loc["Net Debt"].iloc[0]
    return valuation - net_debt


def Price_per_share(equity_value=Equity_Value(), balance_sheet=balance_sheet):
    """
    Prix par action = Equity Value / Nombre d'actions en circulation
    """
    number_of_shares = balance_sheet.loc["Ordinary Shares Number"].iloc[0]
    price_per_share = equity_value / number_of_shares
    return price_per_share


def sensitivity_table(projected_fcf=projected_fcf()):
    """
    Tableau de sensibilité : WACC vs Croissance terminale g
    """
    wacc_range = np.arange(
        float(input("Entrez la valeur minimale de WACC : ")),
        float(input("Entrez la valeur maximale de WACC : ")),
        float(input("Entrez l'incrément pour WACC : "))
    )
    g_range = np.arange(
        float(input("Entrez la valeur minimale de g (croissance terminale) : ")),
        float(input("Entrez la valeur maximale de g (croissance terminale) : ")),
        float(input("Entrez l'incrément pour g (croissance terminale) : "))
    )
    sensibilité_df = pd.DataFrame(index=wacc_range, columns=g_range)
    for wacc in wacc_range:
        for g in g_range:
            if wacc > g:
                dcf = DCF(projected_fcf, wacc)
                tv = projected_fcf.iloc[-1] * (1 + g) / (wacc - g)
                valuation = Final_Valuation(dcf, tv, wacc)
                sensibilité_df.loc[float(wacc), float(g)] = round(float(valuation), 2)
    return sensibilité_df.dropna(axis=0, how="all").sort_index(ascending=False).sort_index(axis=1, ascending=False)


# =========================
# Exécution du modèle
# =========================
fcf_historical = FCF()
fcf_projected = projected_fcf()
Final_FCF = pd.concat([fcf_historical, fcf_projected]).sort_index(ascending=False)

Wacc = WACC()
dcf = DCF()
tv, g = Terminal_Value()
valuation = Final_Valuation()
equity_value = Equity_Value()
price_per_share = Price_per_share()
sensibilité_df = sensitivity_table()

# =========================
# Affichage résultats
# =========================
df1 = pd.DataFrame({"FCF": Final_FCF, "DCF": dcf}).sort_index(ascending=False)
df2 = pd.Series({
    "WACC": Wacc,
    "Terminal Value": tv,
    "Valorisation DCF": valuation,
    "Valeur des capitaux propres": equity_value,
    "Prix par action (estimé)": price_per_share,
    "Prix actuel du marché (réel)": yf.Ticker(ticker).info["currentPrice"]
})

print("---------------------------------------------FCF et DCF------------------------------------------------")
print(df1.applymap(lambda v: f"{v:,.0f} $" if pd.notnull(v) else ""))
print("---------------------------------------------Résultats-------------------------------------------------")
pd.set_option("display.float_format", "{:,.3f}".format)
print(df2)
print("Répartition de la valorisation : DCF = {}%, TV = {}%".format(
    (sum(dcf)/valuation)*100,
    tv/(1+Wacc)**len(dcf)/valuation*100
))
print((
    f"Selon ce modèle, {ticker} semble sous-évalué de {df2['Prix actuel du marché (réel)'] - price_per_share:,.2f} $"
    if price_per_share < df2["Prix actuel du marché (réel)"] 
    else f"Selon ce modèle, {ticker} semble surévalué de {price_per_share - df2['Prix actuel du marché (réel)']:,.2f} $"
))
print("------------------------------------Tableau de sensibilité (WACC et Croissance terminale (g))---------------------------------------")
pd.reset_option("display.float_format")
print(sensibilité_df.applymap(lambda v: f"{v:,.0f} $" if pd.notnull(v) else ""))

# =========================
# Visualisation
# =========================

# FCF historique et projeté
plt.plot(fcf_historical.index, fcf_historical.values, label='FCF Historique', marker='o', color='blue')
plt.plot(fcf_projected.index, fcf_projected.values, label='FCF Projeté', marker='o', color='green', linestyle='--')
plt.title(f"FCF historique et projeté pour {ticker}")
plt.xlabel("Années")
plt.ylabel("Montant en $")
plt.legend()
plt.show()

# FCF vs DCF
df1.plot(kind='bar', figsize=(10, 6), title=f"FCF et DCF pour {ticker}")
plt.xlabel("Années")
plt.ylabel("Montant en $")
plt.legend()
plt.show()

# Répartition de la valorisation : DCF vs Terminal Value
plt.pie(
    [sum(dcf), tv/(1+Wacc)**len(dcf)],
    labels=["DCF", "Valeur Terminale"],
    autopct='%1.1f%%')
plt.title("Répartition de la valorisation")
plt.legend(["DCF", "Valeur Terminale"])
plt.show()

