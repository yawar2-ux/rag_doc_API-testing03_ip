import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, time
import random
import numpy as np
from typing import List, Dict
from psycopg2.extras import DictCursor
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool

def validate_mobile_no(mobile_no):
    """Validate that mobile number is exactly 4 digits"""
    if len(mobile_no) != 4 or not mobile_no.isdigit():
        raise ValueError("Mobile number must be exactly 4 digits")
    return mobile_no


DB_CONFIG = {
    "dbname": "genai",
    "user": "postgres",
    "password": "Translab123",
    "host": "168.138.192.177",
    "port": "5432",
    "connect_timeout": 30,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5
}


class DatabasePool:
    _instance = None
    _pool = None
    _max_retries = 3
    _retry_delay = 1  # seconds

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DatabasePool()
        return cls._instance

    def __init__(self):
        if DatabasePool._pool is None:
            self._initialize_pool()

    def _initialize_pool(self):
        try:
            DatabasePool._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **DB_CONFIG
            )
            logging.info("Database connection pool initialized")
        except Exception as e:
            logging.error(f"Error initializing connection pool: {e}")
            raise

    def _validate_connection(self, conn):
        """Validate if connection is still alive"""
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception:
            return False

    def _get_fresh_connection(self):
        """Get a new connection from the pool with validation"""
        for attempt in range(self._max_retries):
            try:
                conn = DatabasePool._pool.getconn()
                if self._validate_connection(conn):
                    return conn
                else:
                    # Connection is dead, close and try to get a new one
                    DatabasePool._pool.putconn(conn, close=True)
                    self._initialize_pool()
            except Exception as e:
                logging.error(f"Connection attempt {attempt + 1} failed: {e}")
                time.sleep(self._retry_delay)

        raise Exception("Failed to establish database connection after multiple attempts")

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self._get_fresh_connection()
            yield conn
        except psycopg2.OperationalError as e:
            logging.error(f"Operational database error: {e}")
            if conn:
                DatabasePool._pool.putconn(conn, close=True)
            self._initialize_pool()
            raise
        except Exception as e:
            logging.error(f"Error with database connection: {e}")
            raise
        finally:
            if conn:
                try:
                    if not conn.closed:
                        DatabasePool._pool.putconn(conn)
                except Exception as e:
                    logging.error(f"Error returning connection to pool: {e}")

    @classmethod
    def close_all(cls):
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
            logging.info("All database connections closed")

class CallCenterDB:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        # Only create tables if they don't exist
        self.init_tables()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS call_records (
                    id SERIAL PRIMARY KEY,
                    date TIMESTAMP NOT NULL,
                    cust_id VARCHAR(20) NOT NULL,
                    agent_id VARCHAR(10) NOT NULL,
                    mobile_no VARCHAR(4) NOT NULL,
                    call_duration INTERVAL NOT NULL,
                    call_summary TEXT NOT NULL,
                    sentiment FLOAT NOT NULL
                );
            """)
            self.conn.commit()

    def validate_mobile_no(mobile_no):
        if len(mobile_no) != 4 or not mobile_no.isdigit():
            raise ValueError("Mobile number must be exactly 4 digits")
        return mobile_no


    def get_call_history(self, mobile_no):
        """Fetch call history for a given mobile number"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT call_summary
                    FROM call_records
                    WHERE mobile_no = %s
                    ORDER BY date DESC
                    LIMIT 1
                """, (mobile_no,))
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error fetching call history: {e}")
            return None

    def init_tables(self):
        """Initialize tables only if they don't exist"""
        with self.conn.cursor() as cur:
            # Check if tables exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'call_records'
                );
            """)
            tables_exist = cur.fetchone()[0]

            if not tables_exist:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS call_records (
                        id SERIAL PRIMARY KEY,
                        date TIMESTAMP NOT NULL,
                        cust_id VARCHAR(20) NOT NULL,
                        agent_id VARCHAR(10) NOT NULL,
                        call_duration INTERVAL NOT NULL,
                        call_summary TEXT NOT NULL,
                        sentiment FLOAT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS agents (
                        agent_id VARCHAR(10) PRIMARY KEY,
                        last_active TIMESTAMP
                    );
                """)

                # Initialize agents
                for i in range(1, 6):
                    cur.execute(
                        "INSERT INTO agents (agent_id, last_active) VALUES (%s, NULL)",
                        (f"agent_{i}",)
                    )

                self.conn.commit()


    def get_customer_call_summaries(self, mobile_no):
        """Fetch all previous call summaries for a customer."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT call_summary
                    FROM call_records
                    WHERE mobile_no = %s
                    ORDER BY date DESC
                """, (mobile_no,))
                return [record[0] for record in cur.fetchall()]
        except Exception as e:
            self.logger.error(f"Error fetching customer history: {e}")
            return []

    def preprocess_text(self, text):
        """Preprocess text by lowercasing and removing special characters."""
        return re.sub(r'[^\w\s]', '', text.lower().strip())

    def find_similar_in_customer_history(self, current_text: str, mobile_no: str, threshold: float = 0.08):
        """
        Find similar summaries in customer history using TF-IDF and cosine similarity.
        """
        try:
            # Preprocess the current query
            current_text = self.preprocess_text(current_text)
            # print(f"[DEBUG] Preprocessed Query: {current_text}")

            # Fetch previous call summaries for the mobile number
            previous_summaries = self.get_customer_call_summaries(mobile_no)
            # print(f"[DEBUG] Fetched Summaries: {previous_summaries}")

            if not previous_summaries:
                # print("[DEBUG] No previous summaries found.")
                return None

            # Preprocess all previous summaries
            previous_summaries = [self.preprocess_text(summary) for summary in previous_summaries]
            # print(f"[DEBUG] Preprocessed Summaries: {previous_summaries}")

            # Combine the query and previous summaries
            all_texts = [current_text] + previous_summaries

            # TF-IDF vectorization
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            # print(f"[DEBUG] TF-IDF Matrix Shape: {tfidf_matrix.shape}")

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            # print(f"[DEBUG] Similarities: {similarities}")

            # Find the most relevant summary
            max_sim_idx = similarities.argmax()
            max_similarity = similarities[max_sim_idx]
            # print(f"[DEBUG] Max Similarity: {max_similarity}, Index: {max_sim_idx}")

            # Check if similarity exceeds the threshold
            if max_similarity >= threshold:
                return previous_summaries[max_sim_idx]

            return None

        except Exception as e:
            self.logger.error(f"Error in finding similar summary: {e}")
            return None


    def get_all_call_summaries(self, mobile_no: str) -> List[str]:
        """
        Fetch all call summaries for a specific mobile number.
        """
        query = "SELECT call_summary FROM call_records WHERE mobile_no = %s ORDER BY date DESC"
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(query, (mobile_no,))
                return [row["call_summary"] for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error fetching all call summaries: {str(e)}")
            return []



    def save_call_record(self, start_time, cust_id, agent_id, mobile_no, call_transcript, call_summary, avg_sentiment):
        """Save call record to the database."""
        try:
            # Convert avg_sentiment to float if it's a list/array
            if isinstance(avg_sentiment, (list, np.ndarray)):
                avg_sentiment = float(np.mean(avg_sentiment)) if avg_sentiment else 50.0
            else:
                avg_sentiment = float(avg_sentiment) if avg_sentiment is not None else 50.0

            print(f"Saving call record with Avg Sentiment: {avg_sentiment}, Type: {type(avg_sentiment)}")  # Debug print

            end_time = datetime.now()
            call_duration = end_time - start_time

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO call_records
                    (date, cust_id, agent_id, mobile_no, call_duration, call_summary, sentiment)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    start_time,  # This is mapped to the `date` column
                    cust_id,
                    agent_id,
                    mobile_no,
                    call_duration,
                    call_summary,
                    avg_sentiment
                ))
                self.conn.commit()
                print("Call record successfully saved to the database.")
                return True
        except Exception as e:
            self.logger.error(f"Error in save_call_record: {e}")
            print(f"Error saving call record: {e}")
            self.conn.rollback()
            return False

    def select_available_agent(self):
        """Select a random available agent"""
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT agent_id FROM agents
                    WHERE last_active IS NULL
                    OR agent_id NOT IN (
                        SELECT agent_id FROM agents
                        WHERE last_active = (
                            SELECT MAX(last_active) FROM agents
                        )
                    )
                    ORDER BY RANDOM()
                    LIMIT 1
                """)
                agent = cur.fetchone()

                if agent:
                    cur.execute(
                        "UPDATE agents SET last_active = NOW() WHERE agent_id = %s",
                        (agent['agent_id'],)
                    )
                    self.conn.commit()
                    return agent['agent_id']

                # If no available agent, reset all agents and select one
                cur.execute("UPDATE agents SET last_active = NULL")
                cur.execute(
                    "UPDATE agents SET last_active = NOW() WHERE agent_id = 'agent_1' RETURNING agent_id"
                )
                self.conn.commit()
                return 'agent_1'

        except Exception as e:
            self.logger.error(f"Error selecting agent: {e}")
            return 'agent_1'

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

def generate_cust_id():
    """Generate a customer ID"""
    timestamp = datetime.now().strftime('%y%m%d%H%M')  # Using minutes for more uniqueness
    random_num = str(random.randint(0, 999)).zfill(3)  # 3-digit random number
    return f"C{timestamp}{random_num}"

def get_db_connection():
    """Get a database connection instance"""
    try:
        return CallCenterDB()
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def cleanup_db_connections():
    DatabasePool.close_all()