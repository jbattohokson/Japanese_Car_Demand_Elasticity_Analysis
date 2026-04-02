import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "..", "Datasets", "Japanese_Car_Sales_Elasticity.csv")


class CarSalesAnalysis:

    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self._prepare_data()

    def _prepare_data(self):
        self.df.columns = self.df.columns.str.strip()
        self.finance_col = (
            'Finance Rate of New Car (Weighted)'
            if 'Finance Rate of New Car (Weighted)' in self.df.columns
            else None
        )
        print("\nData loaded successfully")
        print(f"  Rows: {len(self.df)}")
        print(f"  Columns: {list(self.df.columns)}")
        self.df['Total_Sales'] = self.df[['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']].sum(axis=1)

    def descriptive_stats(self):
        print("\n" + "="*80)
        print("DESCRIPTIVE STATISTICS")
        print("="*80)
        print("\nSales by Manufacturer (units):")
        print("-"*80)
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        for col in sales_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            print(f"  {col:10} | Mean: {mean:>10,.0f} | Std: {std:>10,.0f} | Min: {min_val:>10,.0f} | Max: {max_val:>10,.0f}")
        print("\nEconomic Indicators:")
        print("-"*80)
        econ_cols = ['CPI', 'Finance Rate of New Car (Weighted)', 'New Car Loan Amt.', 'GDP Growth']
        for col in econ_cols:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                print(f"  {col:35} | Mean: {mean:>10.2f} | Std: {std:>10.2f} | Min: {min_val:>10.2f} | Max: {max_val:>10.2f}")

    def correlation_analysis(self):
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        finance_col = self.finance_col
        gdp_col = 'GDP Growth'
        cpi_col = 'CPI'
        loan_col = 'New Car Loan Amt.'
        print("\nCorrelation with Total Sales:")
        print("-"*80)
        for col, label in [(cpi_col, cpi_col), (loan_col, loan_col), (gdp_col, gdp_col)]:
            if col in self.df.columns:
                corr = self.df[[col, 'Total_Sales']].corr().iloc[0, 1]
                print(f"  {label:35} : {corr:>7.3f}")
        if finance_col:
            print("\nManufacturer Sensitivity to Finance Rate:")
            print("-"*80)
            for mfg in sales_cols:
                corr = self.df[[finance_col, mfg]].corr().iloc[0, 1]
                sensitivity = "High" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.15 else "Low"
                print(f"  {mfg:10} : {corr:>7.3f} ({sensitivity})")
        if gdp_col in self.df.columns:
            print("\nManufacturer Sensitivity to GDP Growth:")
            print("-"*80)
            for mfg in sales_cols:
                corr = self.df[[gdp_col, mfg]].corr().iloc[0, 1]
                sensitivity = "High" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.15 else "Low"
                print(f"  {mfg:10} : {corr:>7.3f} ({sensitivity})")

    def elasticity_analysis(self):
        print("\n" + "="*80)
        print("ELASTICITY ANALYSIS")
        print("="*80)
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        finance_col = self.finance_col
        if finance_col:
            print("\nPrice Elasticity (Finance Rate Impact):")
            print("-"*80)
            median_rate = self.df[finance_col].median()
            for mfg in sales_cols:
                high_rate_sales = self.df[self.df[finance_col] > median_rate][mfg].mean()
                low_rate_sales = self.df[self.df[finance_col] <= median_rate][mfg].mean()
                if low_rate_sales > 0:
                    elasticity = ((high_rate_sales - low_rate_sales) / low_rate_sales) * 100
                    print(f"  {mfg:10} | High Rate Avg: {high_rate_sales:>10,.0f} | Low Rate Avg: {low_rate_sales:>10,.0f} | Elasticity: {elasticity:>+7.1f}%")
        gdp_col = 'GDP Growth'
        if gdp_col in self.df.columns:
            print("\nIncome Elasticity (GDP Growth Impact):")
            print("-"*80)
            for mfg in sales_cols:
                positive_gdp_sales = self.df[self.df[gdp_col] > 0][mfg].mean()
                negative_gdp_sales = self.df[self.df[gdp_col] <= 0][mfg].mean()
                if negative_gdp_sales > 0:
                    elasticity = ((positive_gdp_sales - negative_gdp_sales) / negative_gdp_sales) * 100
                    print(f"  {mfg:10} | Positive GDP Avg: {positive_gdp_sales:>10,.0f} | Negative GDP Avg: {negative_gdp_sales:>10,.0f} | Elasticity: {elasticity:>+7.1f}%")

    def market_analysis(self):
        print("\n" + "="*80)
        print("MARKET ANALYSIS")
        print("="*80)
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        print("\nMarket Share (by total sales):")
        print("-"*80)
        total_by_mfg = self.df[sales_cols].sum()
        grand_total = total_by_mfg.sum()
        for mfg in sales_cols:
            share = (total_by_mfg[mfg] / grand_total) * 100
            print(f"  {mfg:10} : {share:>6.1f}%")
        print("\nAverage Sales by Period:")
        print("-"*80)
        period_col = 'Semi-Annual Year' if 'Semi-Annual Year' in self.df.columns else 'Semi_Annual_Year'
        period_data = self.df[[period_col] + sales_cols].copy()
        print(period_data.to_string(index=False))
        print("\nGrowth Rates (2008-2025):")
        print("-"*80)
        for mfg in sales_cols:
            first = self.df[mfg].iloc[0]
            last = self.df[mfg].iloc[-1]
            growth = ((last - first) / first) * 100
            direction = "up" if growth > 0 else "down"
            print(f"  {mfg:10} : {direction} {growth:>+7.1f}%")

    def plot_sales_trends(self):
        print("\nCreating sales trend visualization...")
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        plt.figure(figsize=(14, 7))
        for col in sales_cols:
            plt.plot(range(len(self.df)), self.df[col], marker='o', label=col, linewidth=2)
        plt.xlabel('Period')
        plt.ylabel('Sales (units)')
        plt.title('Japanese Car Sales Trends (2008-2025)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'sales_trend.png'), dpi=150)
        print("  Saved: sales_trend.png")
        plt.close()

    def plot_market_share(self):
        print("\nCreating market share visualization...")
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        total_by_mfg = self.df[sales_cols].sum()
        grand_total = total_by_mfg.sum()
        shares = (total_by_mfg / grand_total * 100).sort_values()
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(shares.index, shares.values, color='steelblue', edgecolor='white')
        ax.bar_label(bars, fmt='%.1f%%', padding=4)
        ax.set_xlabel('Market Share (%)')
        ax.set_title('Market Share by Manufacturer (2008-2025)')
        ax.set_xlim(0, shares.max() * 1.15)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'market_share.png'), dpi=150)
        print("  Saved: market_share.png")
        plt.close()

    def plot_elasticity_comparison(self):
        print("\nCreating elasticity comparison visualization...")
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        finance_col = self.finance_col
        if not finance_col:
            print("  Finance rate column not found")
            return
        elasticities = [self.df[[finance_col, mfg]].corr().iloc[0, 1] for mfg in sales_cols]
        plt.figure(figsize=(10, 6))
        colors = ['red' if x < 0 else 'green' for x in elasticities]
        plt.barh(sales_cols, elasticities, color=colors)
        plt.axvline(0, color='black', linewidth=1)
        plt.xlabel('Correlation with Finance Rate')
        plt.title('Price Sensitivity by Manufacturer')
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'elasticity_comparison.png'), dpi=150)
        print("  Saved: elasticity_comparison.png")
        plt.close()

    def plot_economic_impact(self):
        print("\nCreating economic impact visualization...")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        sales_cols = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        total_sales = self.df[sales_cols].sum(axis=1)
        if 'CPI' in self.df.columns:
            axes[0, 0].scatter(self.df['CPI'], total_sales, s=100, alpha=0.6)
            axes[0, 0].set_xlabel('CPI')
            axes[0, 0].set_ylabel('Total Sales')
            axes[0, 0].set_title('Sales vs CPI')
            axes[0, 0].grid(True, alpha=0.3)
        finance_col = self.finance_col
        if finance_col:
            axes[0, 1].scatter(self.df[finance_col], total_sales, s=100, alpha=0.6, color='orange')
            axes[0, 1].set_xlabel('Finance Rate (%)')
            axes[0, 1].set_ylabel('Total Sales')
            axes[0, 1].set_title('Sales vs Finance Rate')
            axes[0, 1].grid(True, alpha=0.3)
        if 'GDP Growth' in self.df.columns:
            axes[1, 0].scatter(self.df['GDP Growth'], total_sales, s=100, alpha=0.6, color='green')
            axes[1, 0].set_xlabel('GDP Growth (%)')
            axes[1, 0].set_ylabel('Total Sales')
            axes[1, 0].set_title('Sales vs GDP Growth')
            axes[1, 0].grid(True, alpha=0.3)
        if 'New Car Loan Amt.' in self.df.columns:
            axes[1, 1].scatter(self.df['New Car Loan Amt.'], total_sales, s=100, alpha=0.6, color='red')
            axes[1, 1].set_xlabel('Loan Amount')
            axes[1, 1].set_ylabel('Total Sales')
            axes[1, 1].set_title('Sales vs Loan Availability')
            axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'economic_impact.png'), dpi=150)
        print("  Saved: economic_impact.png")
        plt.close()

    def generate_full_report(self):
        print("\n" + "="*80)
        print("JAPANESE CAR SALES ELASTICITY ANALYSIS")
        print("Complete Report")
        print("="*80)
        self.descriptive_stats()
        self.correlation_analysis()
        self.elasticity_analysis()
        self.market_analysis()
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        self.plot_sales_trends()
        self.plot_market_share()
        self.plot_elasticity_comparison()
        self.plot_economic_impact()
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - sales_trend.png")
        print("  - market_share.png")
        print("  - elasticity_comparison.png")
        print("  - economic_impact.png")


class TableauExporter:

    def __init__(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        self.export_dir = os.path.join(SCRIPT_DIR, "tableau_data")
        os.makedirs(self.export_dir, exist_ok=True)
        print(f"Export directory: {self.export_dir}")
        self._prepare_data()

    def _prepare_data(self):
        self.df.columns = self.df.columns.str.strip()
        self.finance_col = (
            'Finance Rate of New Car (Weighted)'
            if 'Finance Rate of New Car (Weighted)' in self.df.columns
            else None
        )
        print("\nData loaded successfully")
        print(f"  Rows: {len(self.df)}")
        self.df['Total_Sales'] = self.df[['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']].sum(axis=1)

    def export_raw_data(self):
        print("\nExporting Raw Data...")
        df_clean = self.df.copy()
        df_clean.columns = [col.replace('(', '').replace(')', '').replace('.', '').strip()
                            for col in df_clean.columns]
        output_file = os.path.join(self.export_dir, "01_raw_data.csv")
        df_clean.to_csv(output_file, index=False)
        print(f"  Saved: 01_raw_data.csv ({len(df_clean)} rows)")
        return output_file

    def export_long_format(self):
        print("\nExporting Long Format...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        id_vars = [col for col in self.df.columns if col not in manufacturers]
        df_long = self.df.melt(id_vars=id_vars, var_name='Manufacturer', value_name='Sales_Units')
        df_long.columns = [col.replace('(', '').replace(')', '').replace('.', '').strip()
                           for col in df_long.columns]
        output_file = os.path.join(self.export_dir, "02_sales_long_format.csv")
        df_long.to_csv(output_file, index=False)
        print(f"  Saved: 02_sales_long_format.csv ({len(df_long)} rows)")
        return output_file

    def export_market_share(self):
        print("\nExporting Market Share...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        total_by_mfg = self.df[manufacturers].sum()
        grand_total = total_by_mfg.sum()
        market_share_data = [
            {
                'Manufacturer': mfg,
                'Total_Sales': int(total_by_mfg[mfg]),
                'Market_Share_Percent': round((total_by_mfg[mfg] / grand_total) * 100, 2),
                'Market_Share_Decimal': round((total_by_mfg[mfg] / grand_total), 4)
            }
            for mfg in manufacturers
        ]
        df_market = pd.DataFrame(market_share_data)
        output_file = os.path.join(self.export_dir, "03_market_share.csv")
        df_market.to_csv(output_file, index=False)
        print(f"  Saved: 03_market_share.csv ({len(df_market)} rows)")
        return output_file

    def export_elasticity_metrics(self):
        print("\nExporting Elasticity Metrics...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        finance_col = self.finance_col
        gdp_col = 'GDP Growth'
        elasticity_data = []
        if finance_col:
            median_rate = self.df[finance_col].median()
            for mfg in manufacturers:
                high_rate = self.df[self.df[finance_col] > median_rate][mfg].mean()
                low_rate = self.df[self.df[finance_col] <= median_rate][mfg].mean()
                elasticity = ((high_rate - low_rate) / low_rate * 100) if low_rate > 0 else 0
                elasticity_data.append({
                    'Manufacturer': mfg,
                    'Elasticity_Type': 'Price_Elasticity',
                    'High_Rate_Avg_Sales': round(high_rate, 0),
                    'Low_Rate_Avg_Sales': round(low_rate, 0),
                    'Elasticity_Percent': round(elasticity, 2)
                })
        if gdp_col in self.df.columns:
            for mfg in manufacturers:
                positive_gdp = self.df[self.df[gdp_col] > 0][mfg].mean()
                negative_gdp = self.df[self.df[gdp_col] <= 0][mfg].mean()
                elasticity = ((positive_gdp - negative_gdp) / negative_gdp * 100) if negative_gdp > 0 else 0
                elasticity_data.append({
                    'Manufacturer': mfg,
                    'Elasticity_Type': 'Income_Elasticity',
                    'Positive_GDP_Avg_Sales': round(positive_gdp, 0),
                    'Negative_GDP_Avg_Sales': round(negative_gdp, 0),
                    'Elasticity_Percent': round(elasticity, 2)
                })
        df_elasticity = pd.DataFrame(elasticity_data)
        output_file = os.path.join(self.export_dir, "04_elasticity_metrics.csv")
        df_elasticity.to_csv(output_file, index=False)
        print(f"  Saved: 04_elasticity_metrics.csv ({len(df_elasticity)} rows)")
        return output_file

    def export_correlation_matrix(self):
        print("\nExporting Correlation Matrix...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        econ_cols = ['CPI', 'Finance Rate of New Car (Weighted)', 'New Car Loan Amt.', 'GDP Growth']
        econ_cols = [col for col in econ_cols if col in self.df.columns]
        all_cols = manufacturers + econ_cols
        corr_matrix = self.df[all_cols].corr()
        corr_data = [
            {
                'Variable_1': row_name,
                'Variable_2': col_name,
                'Correlation': round(corr_matrix.loc[row_name, col_name], 3)
            }
            for row_name in corr_matrix.index
            for col_name in corr_matrix.columns
        ]
        df_corr = pd.DataFrame(corr_data)
        output_file = os.path.join(self.export_dir, "05_correlation_matrix.csv")
        df_corr.to_csv(output_file, index=False)
        print(f"  Saved: 05_correlation_matrix.csv ({len(df_corr)} rows)")
        return output_file

    def export_growth_rates(self):
        print("\nExporting Growth Rates...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        growth_data = [
            {
                'Manufacturer': mfg,
                'First_Period_Sales': int(self.df[mfg].iloc[0]),
                'Last_Period_Sales': int(self.df[mfg].iloc[-1]),
                'Absolute_Change': int(self.df[mfg].iloc[-1] - self.df[mfg].iloc[0]),
                'Growth_Percent': round(((self.df[mfg].iloc[-1] - self.df[mfg].iloc[0]) / self.df[mfg].iloc[0]) * 100, 2)
            }
            for mfg in manufacturers
        ]
        df_growth = pd.DataFrame(growth_data)
        output_file = os.path.join(self.export_dir, "06_growth_rates.csv")
        df_growth.to_csv(output_file, index=False)
        print(f"  Saved: 06_growth_rates.csv ({len(df_growth)} rows)")
        return output_file

    def export_time_series_analysis(self):
        print("\nExporting Time Series Data...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        period_col = 'Semi-Annual Year'
        finance_col = self.finance_col
        id_vars = [period_col, 'Total_Sales', 'CPI', 'New Car Loan Amt.', 'GDP Growth']
        if finance_col:
            id_vars.append(finance_col)
        id_vars = [col for col in id_vars if col in self.df.columns]
        df_ts = self.df.melt(
            id_vars=id_vars,
            value_vars=manufacturers,
            var_name='Manufacturer',
            value_name='Sales'
        )
        df_ts['Period_Index'] = df_ts.groupby('Manufacturer').cumcount()
        df_ts['Market_Share_Percent'] = (df_ts['Sales'] / df_ts['Total_Sales'] * 100).round(2)
        df_ts['Sales'] = df_ts['Sales'].astype(int)
        df_ts['Total_Market_Sales'] = df_ts['Total_Sales'].astype(int)
        rename_map = {period_col: 'Period', 'New Car Loan Amt.': 'Loan_Amount', 'GDP Growth': 'GDP_Growth'}
        if finance_col:
            rename_map[finance_col] = 'Finance_Rate'
        df_ts = df_ts.rename(columns=rename_map)
        col_order = ['Period', 'Period_Index', 'Manufacturer', 'Sales', 'Total_Market_Sales',
                     'Market_Share_Percent', 'CPI', 'Finance_Rate', 'Loan_Amount', 'GDP_Growth']
        col_order = [c for c in col_order if c in df_ts.columns]
        df_ts = df_ts[col_order].sort_values(['Period_Index', 'Manufacturer']).reset_index(drop=True)
        output_file = os.path.join(self.export_dir, "07_time_series_analysis.csv")
        df_ts.to_csv(output_file, index=False)
        print(f"  Saved: 07_time_series_analysis.csv ({len(df_ts)} rows)")
        return output_file

    def export_summary_statistics(self):
        print("\nExporting Summary Statistics...")
        manufacturers = ['Toyota', 'Honda', 'Mazda', 'Nissan', 'Subaru']
        econ_cols = ['CPI', 'Finance Rate of New Car (Weighted)', 'New Car Loan Amt.', 'GDP Growth']
        econ_cols = [col for col in econ_cols if col in self.df.columns]
        summary_data = [
            {
                'Variable': col,
                'Variable_Type': 'Sales' if col in manufacturers else 'Economic',
                'Mean': round(self.df[col].mean(), 0 if col in manufacturers else 2),
                'Std_Dev': round(self.df[col].std(), 0 if col in manufacturers else 2),
                'Min': round(self.df[col].min(), 0 if col in manufacturers else 2),
                'Max': round(self.df[col].max(), 0 if col in manufacturers else 2),
                'Median': round(self.df[col].median(), 0 if col in manufacturers else 2)
            }
            for col in manufacturers + econ_cols
        ]
        df_summary = pd.DataFrame(summary_data)
        output_file = os.path.join(self.export_dir, "08_summary_statistics.csv")
        df_summary.to_csv(output_file, index=False)
        print(f"  Saved: 08_summary_statistics.csv ({len(df_summary)} rows)")
        return output_file

    def export_all(self):
        print("\n" + "="*80)
        print("TABLEAU DATA EXPORT")
        print("="*80)
        files = [
            self.export_raw_data(),
            self.export_long_format(),
            self.export_market_share(),
            self.export_elasticity_metrics(),
            self.export_correlation_matrix(),
            self.export_growth_rates(),
            self.export_time_series_analysis(),
            self.export_summary_statistics(),
        ]
        print("\n" + "="*80)
        print("EXPORT COMPLETE")
        print("="*80)
        print(f"\nAll files saved to: {self.export_dir}")
        print("\nFiles created:")
        print("  1. 01_raw_data.csv - Original data cleaned")
        print("  2. 02_sales_long_format.csv - Long format for line charts")
        print("  3. 03_market_share.csv - Market share percentages")
        print("  4. 04_elasticity_metrics.csv - Price & income elasticity")
        print("  5. 05_correlation_matrix.csv - Variable correlations")
        print("  6. 06_growth_rates.csv - Growth rates by manufacturer")
        print("  7. 07_time_series_analysis.csv - Complete time series")
        print("  8. 08_summary_statistics.csv - Descriptive statistics")
        return files


if __name__ == "__main__":
    try:
        print("="*80)
        print("STEP 1: Running analysis...")
        print("="*80)
        analysis = CarSalesAnalysis(CSV_FILE)
        analysis.generate_full_report()

        print("\n" + "="*80)
        print("STEP 2: Exporting Tableau data...")
        print("="*80)
        exporter = TableauExporter(CSV_FILE)
        exporter.export_all()

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"\nMake sure the CSV file is in: {os.path.join(SCRIPT_DIR, '..', 'Datasets')}")
        print(f"File name should be: Japanese_Car_Sales_Elasticity.csv")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nPlease check:")
        print("  1. CSV file exists and is readable")
        print("  2. All required packages are installed")
        print("  3. Column names match the CSV")