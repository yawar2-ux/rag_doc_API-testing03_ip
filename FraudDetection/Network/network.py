import pandas as pd
import networkx as nx
import time
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse

# Create FastAPI app
router = APIRouter()


def load_fraud_data(file_obj):
    """
    Load CSV data and validate it has the required is_fraud column.

    Args:
        file_obj: File-like object containing CSV data

    Returns:
        pandas.DataFrame: Loaded data or None if error
    """
    try:
        df = pd.read_csv(file_obj)
        # Validate that is_fraud column exists
        if 'is_fraud' not in df.columns:
            raise ValueError("Dataset must contain 'is_fraud' column")

        print(f"Successfully loaded data with {len(df)} records and {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def analyze_dataset_columns(df):
    """
    Analyze the columns in the dataset and categorize them.
    Only 'is_fraud' is required.

    Args:
        df: DataFrame containing transaction data

    Returns:
        dict: Information about dataset columns
    """
    # Prepare column information
    column_info = {
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "column_types": {}
    }

    # Analyze each column
    for col in df.columns:
        col_type = str(df[col].dtype)
        column_info["column_types"][col] = col_type

        # Skip is_fraud column as it's special
        if col == 'is_fraud':
            continue

        # Categorize column based on data type
        if pd.api.types.is_numeric_dtype(df[col]):
            column_info["numeric_columns"].append(col)
        elif pd.api.types.is_datetime64_dtype(df[col]) or 'datetime' in col_type:
            column_info["datetime_columns"].append(col)
        else:
            # Try to convert non-numeric columns to datetime
            try:
                pd.to_datetime(df[col])
                column_info["datetime_columns"].append(col)
            except:
                # If not datetime, treat as categorical
                column_info["categorical_columns"].append(col)

    # Add some basic statistics for each column
    column_info["column_stats"] = {}
    for col in df.columns:
        if col in column_info["numeric_columns"] or col == 'is_fraud':
            column_info["column_stats"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "fraud_correlation": float(df[col].corr(df['is_fraud']))
            }
        elif col in column_info["categorical_columns"]:
            # For categorical, give value counts (top 5)
            value_counts = df[col].value_counts().head(5).to_dict()
            column_info["column_stats"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": value_counts
            }

    return column_info

def create_fraud_network(df, column_info, max_nodes=1000):
    """
    Create a network graph from transaction data using any available columns.
    Only requires 'is_fraud' column.

    Args:
        df: DataFrame containing transaction data
        column_info: Dictionary with column analysis information
        max_nodes: Maximum number of nodes to include in the graph

    Returns:
        networkx.Graph: Graph representing transaction network
    """
    start_time = time.time()

    # Sample data if needed
    if len(df) > max_nodes:
        print(f"Dataset is large. Sampling {max_nodes} transactions...")

        # Stratified sampling to maintain fraud/non-fraud ratio
        fraud_df = df[df['is_fraud'] == 1]
        non_fraud_df = df[df['is_fraud'] == 0]

        # Calculate proportions to maintain
        fraud_ratio = len(fraud_df) / len(df)
        fraud_sample_size = int(max_nodes * fraud_ratio)
        non_fraud_sample_size = max_nodes - fraud_sample_size

        # Sample from each group
        if len(fraud_df) > fraud_sample_size:
            fraud_sample = fraud_df.sample(fraud_sample_size, random_state=42)
        else:
            fraud_sample = fraud_df

        if len(non_fraud_df) > non_fraud_sample_size:
            non_fraud_sample = non_fraud_df.sample(non_fraud_sample_size, random_state=42)
        else:
            non_fraud_sample = non_fraud_df

        # Combine samples
        df_sample = pd.concat([fraud_sample, non_fraud_sample])
        df_sample = df_sample.reset_index(drop=True)
    else:
        df_sample = df

    # Create a graph
    G = nx.Graph()

    # Add nodes for each transaction with all properties from the dataset
    print("Adding nodes to the network...")
    for idx, row in df_sample.iterrows():
        node_id = f"TX_{idx}"

        # Add all columns as node attributes
        node_attrs = {}
        for col in df_sample.columns:
            # Handle different data types appropriately
            if col in column_info["numeric_columns"] or col == 'is_fraud':
                # Store numeric values as floats
                node_attrs[col] = float(row[col])
            elif col in column_info["datetime_columns"]:
                # Store datetime as strings
                node_attrs[col] = str(row[col])
            else:
                # Store other values as strings
                node_attrs[col] = str(row[col])

        G.add_node(node_id, **node_attrs)

    # Select columns for grouping
    # Prefer categorical columns with fewer unique values
    grouping_cols = []
    categorical_cols = column_info["categorical_columns"]
    if categorical_cols:
        # Sort categorical columns by number of unique values (ascending)
        sorted_cat_cols = sorted(
            categorical_cols,
            key=lambda col: df_sample[col].nunique()
        )
        # Take up to 2 categorical columns with fewest unique values
        grouping_cols = sorted_cat_cols[:min(2, len(sorted_cat_cols))]

    # If no suitable categorical columns, use binned versions of numeric columns
    if not grouping_cols and column_info["numeric_columns"]:
        # Find numeric columns with highest correlation to fraud
        corr_columns = []
        for col in column_info["numeric_columns"]:
            if col in column_info["column_stats"]:
                corr_columns.append((col, abs(column_info["column_stats"][col]["fraud_correlation"])))

        # Sort by correlation (descending)
        corr_columns.sort(key=lambda x: x[1], reverse=True)

        # Use top correlated column for binning if it exists
        if corr_columns:
            top_col = corr_columns[0][0]
            # Create a binned version for grouping
            df_sample[f"{top_col}_bin"] = pd.qcut(
                df_sample[top_col],
                5,
                labels=False,
                duplicates='drop'
            )
            grouping_cols.append(f"{top_col}_bin")

    # Create edges based on similarities
    print("Creating edges between similar transactions...")

    if grouping_cols:
        # Group transactions by selected columns
        for group_values, group_df in df_sample.groupby(grouping_cols):
            if len(group_df) <= 1:
                continue

            # Connect transactions within the same group
            indices = group_df.index.tolist()
            for i in range(len(indices)):
                for j in range(i+1, min(i+15, len(indices))):  # Limit connections
                    idx1, idx2 = indices[i], indices[j]
                    row1, row2 = group_df.iloc[i], group_df.iloc[j]

                    # Calculate similarity based on numeric columns
                    similarity = 0
                    for col in column_info["numeric_columns"]:
                        try:
                            val1 = float(row1[col])
                            val2 = float(row2[col])

                            # Skip columns with zero values
                            if val1 == 0 or val2 == 0:
                                continue

                            # Calculate relative difference
                            max_val = max(abs(val1), abs(val2))
                            if max_val > 0:  # Avoid division by zero
                                relative_diff = abs(val1 - val2) / max_val
                                # Consider similar if within 20%
                                if relative_diff < 0.2:
                                    similarity += 1
                        except (ValueError, TypeError):
                            # Skip if conversion fails
                            continue

                    # Add edge if similarity is high enough
                    if similarity > 0:
                        node1 = f"TX_{idx1}"
                        node2 = f"TX_{idx2}"
                        weight = 1 + similarity * 0.2
                        G.add_edge(node1, node2, weight=weight)
    else:
        # Without categorical columns, use numeric similarity
        numeric_cols = column_info["numeric_columns"]
        # Skip if no numeric columns
        if not numeric_cols:
            print("No suitable columns for creating connections. Graph will have no edges.")
        else:
            for i in range(len(df_sample)):
                # Connect to a limited number of transactions
                for j in range(i+1, min(i+50, len(df_sample))):
                    row1 = df_sample.iloc[i]
                    row2 = df_sample.iloc[j]

                    similarity = 0
                    for col in numeric_cols:
                        try:
                            val1 = float(row1[col])
                            val2 = float(row2[col])

                            # Skip columns with zero values
                            if val1 == 0 or val2 == 0:
                                continue

                            # Calculate relative difference
                            max_val = max(abs(val1), abs(val2))
                            if max_val > 0:  # Avoid division by zero
                                relative_diff = abs(val1 - val2) / max_val
                                # Consider similar if within 15% (stricter threshold)
                                if relative_diff < 0.15:
                                    similarity += 1
                        except (ValueError, TypeError):
                            # Skip if conversion fails
                            continue

                    # Add edge if similarity is high enough
                    if similarity > 2:  # Higher threshold for non-grouped connections
                        node1 = f"TX_{i}"
                        node2 = f"TX_{j}"
                        weight = 1 + similarity * 0.2
                        G.add_edge(node1, node2, weight=weight)

    # Connect fraud transactions with additional edges
    fraud_indices = df_sample[df_sample['is_fraud'] == 1].index.tolist()

    for i in range(len(fraud_indices)):
        for j in range(i+1, min(i+10, len(fraud_indices))):  # Limit connections
            idx1, idx2 = fraud_indices[i], fraud_indices[j]
            row1 = df_sample.loc[idx1]
            row2 = df_sample.loc[idx2]

            # Calculate similarity between fraud transactions
            similarity = 0
            for col in column_info["numeric_columns"]:
                try:
                    val1 = float(row1[col])
                    val2 = float(row2[col])

                    # Skip columns with zero values
                    if val1 == 0 or val2 == 0:
                        continue

                    # Calculate relative difference
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:  # Avoid division by zero
                        relative_diff = abs(val1 - val2) / max_val
                        # More lenient threshold for fraud-fraud connections
                        if relative_diff < 0.3:
                            similarity += 1
                except (ValueError, TypeError):
                    # Skip if conversion fails
                    continue

            # Add edge if similarity is high enough
            if similarity > 1:
                node1 = f"TX_{idx1}"
                node2 = f"TX_{idx2}"
                weight = 1 + similarity * 0.2
                G.add_edge(node1, node2, weight=weight)

    end_time = time.time()
    print(f"Network creation took {end_time - start_time:.2f} seconds")
    return G

def graph_to_json(G):
    """
    Convert NetworkX graph to JSON format for frontend visualization.
    Preserves all node attributes from the original dataset.

    Args:
        G: NetworkX graph object

    Returns:
        dict: JSON-serializable graph representation
    """
    # Extract node data with all attributes
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        # Create a node dictionary with id and all attributes
        node_dict = {'id': node_id}
        # Add all node attributes - this preserves ALL columns from the original dataset
        for attr, value in node_data.items():
            # Handle numeric values properly
            if isinstance(value, (int, float)):
                node_dict[attr] = float(value)
            else:
                node_dict[attr] = value
        nodes.append(node_dict)

    # Extract edge data with weights
    links = []
    for source, target, data in G.edges(data=True):
        links.append({
            "source": source,
            "target": target,
            "weight": float(data.get('weight', 1.0))
        })

    return {
        "nodes": nodes,
        "links": links
    }

def analyze_network(G):
    """
    Calculate network metrics to identify important nodes and communities.
    Works with any graph structure, only relies on is_fraud node attribute.

    Args:
        G: NetworkX graph object

    Returns:
        dict: Analysis results including high centrality nodes and communities
    """
    # Calculate various centrality measures
    print("Calculating network metrics...")
    degree_centrality = nx.degree_centrality(G)

    # Only calculate betweenness for smaller networks (computationally expensive)
    if len(G.nodes()) <= 1000:
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes())), normalized=True)
    else:
        print("Network too large for betweenness centrality calculation. Skipping...")
        betweenness_centrality = {}

    # Get fraud status
    fraud_status = nx.get_node_attributes(G, 'is_fraud')

    # Find nodes with high centrality
    high_centrality_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    centrality_results = []

    for node, centrality in high_centrality_nodes:
        is_fraud = bool(fraud_status.get(node, 0) == 1)

        # Get all node properties to include in results
        node_props = G.nodes[node].copy()
        if 'is_fraud' in node_props:
            del node_props['is_fraud']  # Remove since we're already including it

        centrality_results.append({
            "node": node,
            "centrality": float(centrality),
            "is_fraud": is_fraud,
            "properties": node_props
        })

    # Community detection results
    community_results = []
    node_communities = {}

    # Optional: Community detection
    try:
        # Try to detect communities if python-louvain is installed
        from community import best_partition

        # Only perform community detection on smaller networks
        if len(G.nodes()) <= 2000:
            partition = best_partition(G)
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)

            # Find the largest communities
            sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            for community_id, nodes in sorted_communities[:5]:  # Show top 5 communities
                if len(nodes) > 1:  # Only show communities with multiple nodes
                    fraud_count = sum(1 for node in nodes if fraud_status.get(node, 0) == 1)
                    community_results.append({
                        "community_id": community_id,
                        "size": len(nodes),
                        "fraud_count": fraud_count,
                        "fraud_ratio": float(fraud_count / len(nodes)),
                        "sample_nodes": nodes[:5]  # Include some sample nodes
                    })

            # Add community ID to each node
            for node, community_id in partition.items():
                node_communities[node] = community_id
    except ImportError:
        print("Community detection requires python-louvain package.")

    # Return the analysis results
    return {
        "high_centrality_nodes": centrality_results,
        "communities": community_results,
        "node_communities": node_communities
    }

@router.post("/network", response_class=JSONResponse)
async def analyze_fraud_network(file: UploadFile = File(...)):
    """
    Upload a CSV file with transaction data and get a fraud network analysis.

    Required column:
    - is_fraud: Binary indicator (1 for fraud, 0 for non-fraud)

    All other columns will be automatically used for analysis.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        # Load the dataset from the uploaded file
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        df = load_fraud_data(file_obj)

        if df is None:
            raise HTTPException(status_code=400, detail="Error processing the uploaded file")

        # Analyze dataset columns
        column_info = analyze_dataset_columns(df)

        # Basic dataset statistics
        stats = {
            "total_transactions": len(df),
            "fraud_transactions": int(df['is_fraud'].sum()),
            "fraud_percentage": float(df['is_fraud'].mean() * 100),
            "column_count": len(df.columns)
        }

        # Find columns with highest correlation to fraud
        if column_info["numeric_columns"]:
            # Get correlations for numeric columns
            correlations = []
            for col in column_info["numeric_columns"]:
                if col in column_info["column_stats"]:
                    correlations.append({
                        "column": col,
                        "correlation": column_info["column_stats"][col]["fraud_correlation"]
                    })

            # Sort by absolute correlation value (descending)
            correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            # Include top correlations in stats
            stats["top_fraud_correlations"] = correlations[:5]

        # Create the network using dynamic column information
        G = create_fraud_network(df, column_info, max_nodes=1000)

        # Network stats
        network_stats = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }

        # Analyze the network
        analysis_results = analyze_network(G)

        # Extract subgraph of fraud nodes to analyze fraud network characteristics
        fraud_nodes = [n for n, d in G.nodes(data=True) if d.get('is_fraud', 0) == 1]
        if fraud_nodes:
            fraud_subgraph = G.subgraph(fraud_nodes)
            network_stats["fraud_subgraph"] = {
                "node_count": fraud_subgraph.number_of_nodes(),
                "edge_count": fraud_subgraph.number_of_edges(),
                "density": nx.density(fraud_subgraph) if fraud_subgraph.number_of_nodes() > 1 else 0,
                "avg_degree": sum(dict(fraud_subgraph.degree()).values()) / fraud_subgraph.number_of_nodes() if fraud_subgraph.number_of_nodes() > 0 else 0
            }

            # Add connected components analysis for fraud subgraph
            connected_components = list(nx.connected_components(fraud_subgraph))
            network_stats["fraud_subgraph"]["connected_components"] = len(connected_components)
            if connected_components:
                largest_cc = max(connected_components, key=len)
                network_stats["fraud_subgraph"]["largest_component_size"] = len(largest_cc)
                network_stats["fraud_subgraph"]["largest_component_percentage"] = len(largest_cc) / len(fraud_nodes) * 100 if len(fraud_nodes) > 0 else 0

        # Convert the graph to JSON format for frontend visualization
        graph_data = graph_to_json(G)

        # Add community information to nodes if available
        if analysis_results.get("node_communities"):
            for node in graph_data["nodes"]:
                node_id = node["id"]
                if node_id in analysis_results["node_communities"]:
                    node["community"] = analysis_results["node_communities"][node_id]

        # Return the results with column information
        return {
            "dataset_stats": stats,
            "network_stats": network_stats,
            "analysis": {
                "high_centrality_nodes": analysis_results["high_centrality_nodes"],
                "communities": analysis_results["communities"]
            },
            "graph_data": graph_data,
            "column_info": column_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Add a CSV schema validation endpoint
@router.post("/validate-schema/", response_class=JSONResponse)
async def validate_schema(file: UploadFile = File(...)):
    """
    Validate that a CSV file has the required schema (is_fraud column)
    and return information about detected columns.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        # Load just the header to validate schema
        contents = await file.read(1024 * 10)  # Read first 10KB to get header
        file_obj = io.BytesIO(contents)
        df_sample = pd.read_csv(file_obj, nrows=5)  # Read just a few rows

        # Check for required is_fraud column
        if 'is_fraud' not in df_sample.columns:
            return {
                "valid": False,
                "message": "Required column 'is_fraud' not found in CSV file",
                "columns": list(df_sample.columns)
            }

        # Analyze the columns
        column_info = analyze_dataset_columns(df_sample)

        return {
            "valid": True,
            "message": "CSV schema is valid for fraud analysis",
            "columns": list(df_sample.columns),
            "column_info": column_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating file: {str(e)}")


