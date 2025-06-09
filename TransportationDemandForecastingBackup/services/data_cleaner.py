import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings('ignore')

class SimpleTimeSeriesCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_lag_features(self, df, target_col='ridership', lags=[1, 7]):
        """Create simple lag features"""
        if 'datetime' in df.columns and 'zone' in df.columns:
            df = df.sort_values(['zone', 'datetime'])
            
            for lag in lags:
                df[f'{target_col}_lag_{lag}'] = df.groupby('zone')[target_col].shift(lag)
            
            # Create 7-day rolling average
            df[f'{target_col}_rolling_7'] = df.groupby('zone')[target_col].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def create_time_features(self, df):
        """Create basic time features"""
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Extract basic time features if they don't exist
            if 'hour' not in df.columns:
                df['hour'] = df['datetime'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['datetime'].dt.dayofweek
            if 'month' not in df.columns:
                df['month'] = df['datetime'].dt.month
            
            # Cyclical features for better modeling
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def handle_missing_and_encode(self, df, target_col='ridership'):
        """Handle missing values and encode categoricals"""
        # Handle categorical encoding
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'datetime':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # Use modern pandas methods for filling missing values
        df = df.ffill().bfill()
        
        # Fill any remaining missing values with 0
        df = df.fillna(0)
        
        return df
    
    def scale_features(self, df, target_col='ridership'):
        """Scale numerical features except target"""
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_col]
        
        if feature_cols:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df
    
    def select_top_features(self, df, target_col='ridership', k=15):
        """Select top K correlated features"""
        if target_col not in df.columns:
            # If target column doesn't exist, just return the dataframe
            return df
            
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != target_col]
        
        if not feature_cols:
            return df
        
        # Calculate correlations and select top K
        correlations = abs(df[feature_cols].corrwith(df[target_col]))
        top_features = correlations.nlargest(min(k, len(feature_cols))).index.tolist()
        
        # Keep essential columns
        keep_cols = []
        for col in ['datetime', 'zone', 'location']:
            if col in df.columns:
                keep_cols.append(col)
        
        keep_cols.extend(top_features)
        keep_cols.append(target_col)
        
        # Remove duplicates while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))
        keep_cols = [col for col in keep_cols if col in df.columns]
        
        return df[keep_cols]
    
    def clean_data(self, df):
        """Main cleaning function that takes a DataFrame and returns cleaned DataFrame"""
        print("🧹 Starting simple data cleaning...")
        print(f"📊 Input data shape: {df.shape}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # 1. Create time features
        df = self.create_time_features(df)
        print("✅ Created time features")
        
        # 2. Create lag features (only if ridership column exists)
        if 'ridership' in df.columns:
            df = self.create_lag_features(df)
            print("✅ Created lag features")
        
        # 3. Handle missing values and encode
        df = self.handle_missing_and_encode(df)
        print("✅ Handled missing values and encoding")
        
        # 4. Scale features
        df = self.scale_features(df)
        print("✅ Scaled features")
        
        # 5. Select top features (only if ridership column exists)
        if 'ridership' in df.columns:
            df = self.select_top_features(df, k=15)
            print("✅ Selected top 15 features")
        
        print(f"📈 Final shape: {df.shape}")
        
        return df

def main():
    """Run the simplified cleaner"""
    input_file = "transportation_time_based_dataset.csv"
    output_file = "cleaned_transportation_data.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found. Run dataset generator first.")
        return
    
    cleaner = SimpleTimeSeriesCleaner()
    df = pd.read_csv(input_file)
    df_clean = cleaner.clean_data(df)
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n🎉 Cleaning complete! Saved to {output_file}")
    print("🚀 Ready for ML modeling.")

if __name__ == "__main__":
    main()