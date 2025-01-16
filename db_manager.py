import sqlite3
import time

DB_FILE = "new_user_profiles.db"

def get_connection(timeout=10):
    """
    Establish a connection to the SQLite database.
    Uses a timeout to handle potential database locks.
    """
    return sqlite3.connect(DB_FILE, timeout=timeout)

def create_table():
    """
    Creates the new_user_profiles table if it does not exist.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS new_user_profiles (
            username TEXT NOT NULL,
            user_id TEXT PRIMARY KEY,
            preferred_province TEXT,
            category_of_interest TEXT,
            activity_level INTEGER
        )
        """)
        conn.commit()

def insert_user(username, user_id, preferred_province, category_of_interest, activity_level):
    """
    Inserts a new user into the new_user_profiles table.
    Implements retries to handle potential database locks.
    """
    for attempt in range(5):  # Retry up to 5 times
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO new_user_profiles (username, user_id, preferred_province, category_of_interest, activity_level)
                VALUES (?, ?, ?, ?, ?)
                """, (username, user_id, preferred_province, category_of_interest, activity_level))
                conn.commit()
                return  # Exit function if insertion is successful
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                raise  # Re-raise non-lock-related errors
    raise Exception("Failed to insert user after multiple attempts due to database lock.")

def fetch_all_users():
    """
    Fetches all users from the new_user_profiles table.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM new_user_profiles")
        return cursor.fetchall()

def fetch_user_by_id(user_id):
    """
    Fetches a user by user_id from the new_user_profiles table.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM new_user_profiles WHERE user_id = ?", (user_id,))
        return cursor.fetchone()
