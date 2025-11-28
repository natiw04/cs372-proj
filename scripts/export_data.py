#!/usr/bin/env python3
"""
Export data from PocketBase for the Tribly AI Assistant.

This script connects to a running PocketBase instance and exports
sample data for reviews, events, posts, classes, teachers, and resources.
All data is anonymized to remove PII.

Usage:
    python scripts/export_data.py             # Export from PocketBase or create sample data
    python scripts/export_data.py --sample    # Create minimal sample data (18 records)
    python scripts/export_data.py --generate  # Generate ~100 synthetic records for testing

Environment Variables:
    POCKETBASE_URL: Base URL of PocketBase (default: http://localhost:8090)
"""

import os
import sys
import json
import hashlib
import asyncio
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Optional imports - not needed for sample data generation
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback for tqdm
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Configuration
POCKETBASE_URL = os.getenv('POCKETBASE_URL', 'http://localhost:8090')
OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'raw'
MAX_RECORDS_PER_COLLECTION = 1000  # Limit for sample data

# Collections to export with their configurations
COLLECTIONS_CONFIG = {
    'group_reviews': {
        'fields': ['id', 'group', 'author', 'rating', 'comment', 'rating_fields',
                   'is_anonymous', 'helpful_count', 'created', 'updated'],
        'expand': 'group',
        'anonymize': ['author'],
        'output_file': 'reviews.json'
    },
    'hangouts': {
        'fields': ['id', 'title', 'description', 'category', 'tags', 'location',
                   'date_time', 'creator', 'attendees', 'everyone', 'created', 'updated'],
        'expand': '',
        'anonymize': ['creator', 'attendees'],
        'output_file': 'hangouts.json'
    },
    'org_events': {
        'fields': ['id', 'title', 'description', 'location', 'date_time', 'duration',
                   'host_group', 'attendees', 'free_food', 'tags', 'created', 'updated'],
        'expand': 'host_group',
        'anonymize': ['attendees'],
        'output_file': 'events.json'
    },
    'feed_posts': {
        'fields': ['id', 'title', 'text_content', 'category', 'tags', 'author',
                   'upvotes', 'downvotes', 'comment_count', 'is_anonymous',
                   'created', 'updated'],
        'expand': '',
        'anonymize': ['author'],
        'output_file': 'posts.json'
    },
    'classes': {
        'fields': ['id', 'title', 'description', 'teachers', 'majors',
                   'avg_rating', 'review_count', 'created', 'updated'],
        'expand': 'teachers',
        'anonymize': [],
        'output_file': 'classes.json'
    },
    'teachers': {
        'fields': ['id', 'name', 'bio', 'research_description', 'classes',
                   'majors', 'avg_rating', 'review_count', 'created', 'updated'],
        'expand': '',
        'anonymize': [],  # Teacher names are public info
        'output_file': 'teachers.json'
    },
    'resources': {
        'fields': ['id', 'title', 'description', 'type', 'tags', 'group',
                   'author', 'upvotes', 'downvotes', 'created', 'updated'],
        'expand': 'group',
        'anonymize': ['author'],
        'output_file': 'resources.json'
    },
    'groups': {
        'fields': ['id', 'name', 'type', 'description', 'tags',
                   'created', 'updated'],
        'expand': '',
        'anonymize': [],
        'output_file': 'groups.json'
    }
}

# User ID anonymization cache
user_id_map: Dict[str, str] = {}
user_counter = 0


def anonymize_user_id(original_id: str) -> str:
    """Convert a real user ID to an anonymous ID."""
    global user_counter

    if not original_id:
        return ''

    if original_id not in user_id_map:
        user_counter += 1
        # Create a deterministic but anonymized ID
        hash_prefix = hashlib.md5(original_id.encode()).hexdigest()[:8]
        user_id_map[original_id] = f"user_{user_counter:04d}_{hash_prefix}"

    return user_id_map[original_id]


def anonymize_user_list(original_ids: List[str]) -> List[str]:
    """Anonymize a list of user IDs."""
    if not original_ids:
        return []
    return [anonymize_user_id(uid) for uid in original_ids]


def process_record(record: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single record, extracting fields and anonymizing as needed."""
    processed = {}

    for field in config['fields']:
        if field in record:
            value = record[field]

            # Anonymize if needed
            if field in config['anonymize']:
                if isinstance(value, list):
                    value = anonymize_user_list(value)
                elif isinstance(value, str):
                    value = anonymize_user_id(value)

            processed[field] = value

    # Handle expanded relations
    if 'expand' in record and record['expand']:
        processed['_expanded'] = {}
        for key, expanded_data in record['expand'].items():
            if isinstance(expanded_data, dict):
                # Single relation
                processed['_expanded'][key] = {
                    'id': expanded_data.get('id'),
                    'name': expanded_data.get('name') or expanded_data.get('title'),
                    'type': expanded_data.get('type')
                }
            elif isinstance(expanded_data, list):
                # Multiple relations
                processed['_expanded'][key] = [
                    {
                        'id': item.get('id'),
                        'name': item.get('name') or item.get('title'),
                        'type': item.get('type')
                    }
                    for item in expanded_data
                ]

    return processed


async def fetch_collection(
    client: Any,  # httpx.AsyncClient when available
    collection: str,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Fetch all records from a collection."""
    records = []
    page = 1
    per_page = 100

    print(f"\nFetching {collection}...")

    while len(records) < MAX_RECORDS_PER_COLLECTION:
        try:
            params = {
                'page': page,
                'perPage': per_page,
                'sort': '-created'  # Most recent first
            }

            if config.get('expand'):
                params['expand'] = config['expand']

            response = await client.get(
                f"{POCKETBASE_URL}/api/collections/{collection}/records",
                params=params,
                timeout=30.0
            )

            if response.status_code == 404:
                print(f"  Collection '{collection}' not found, skipping...")
                return []

            response.raise_for_status()
            data = response.json()

            items = data.get('items', [])
            if not items:
                break

            for item in items:
                processed = process_record(item, config)
                records.append(processed)

                if len(records) >= MAX_RECORDS_PER_COLLECTION:
                    break

            total_pages = data.get('totalPages', 1)
            if page >= total_pages:
                break

            page += 1

        except httpx.HTTPStatusError as e:
            print(f"  HTTP error fetching {collection}: {e}")
            break
        except Exception as e:
            print(f"  Error fetching {collection}: {e}")
            break

    print(f"  Fetched {len(records)} records from {collection}")
    return records


async def export_all_collections():
    """Export all configured collections."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting data from {POCKETBASE_URL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 50)

    async with httpx.AsyncClient() as client:
        # Test connection
        try:
            response = await client.get(f"{POCKETBASE_URL}/api/health", timeout=5.0)
            print(f"PocketBase health check: {response.status_code}")
        except Exception as e:
            print(f"Warning: Could not reach PocketBase at {POCKETBASE_URL}")
            print(f"Error: {e}")
            print("\nMake sure PocketBase is running and the URL is correct.")
            return

        # Export each collection
        export_stats = {}

        for collection, config in COLLECTIONS_CONFIG.items():
            records = await fetch_collection(client, collection, config)

            if records:
                output_file = OUTPUT_DIR / config['output_file']
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=2, ensure_ascii=False, default=str)

                export_stats[collection] = len(records)
                print(f"  Saved to {output_file}")

    # Print summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)

    total_records = 0
    for collection, count in export_stats.items():
        print(f"  {collection}: {count} records")
        total_records += count

    print(f"\nTotal records exported: {total_records}")
    print(f"Unique users anonymized: {len(user_id_map)}")

    # Save user ID mapping (for debugging, not for production)
    mapping_file = OUTPUT_DIR / '_user_id_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(user_id_map, f, indent=2)
    print(f"\nUser ID mapping saved to {mapping_file}")
    print("(This file should NOT be included in the final submission)")

    # Create .gitkeep files to preserve directory structure
    for subdir in ['processed', 'evaluation']:
        gitkeep = OUTPUT_DIR.parent / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()

    # Also create .gitkeep in results subdirs
    for subdir in ['metrics', 'figures', 'ablation']:
        gitkeep = OUTPUT_DIR.parent.parent / 'results' / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()


def create_sample_data():
    """Create sample data if PocketBase is not available."""
    print("Creating sample data (PocketBase not available)...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sample reviews
    sample_reviews = [
        {
            "id": "review_001",
            "group": "class_cs201",
            "author": "user_0001",
            "rating": 4.5,
            "comment": "Great professor! Explains concepts clearly and is always willing to help during office hours. The workload is manageable if you stay on top of assignments.",
            "rating_fields": {"clarity": 5, "helpfulness": 4, "workload": 3},
            "is_anonymous": False,
            "helpful_count": 12,
            "created": "2024-09-15T10:30:00Z",
            "_expanded": {"group": {"id": "class_cs201", "name": "Introduction to Computer Science", "type": "class"}}
        },
        {
            "id": "review_002",
            "group": "teacher_smith",
            "author": "user_0002",
            "rating": 3.0,
            "comment": "Lectures can be dry but the material is important. Make sure to do the readings before class.",
            "rating_fields": {"clarity": 3, "helpfulness": 3, "fairness": 4},
            "is_anonymous": True,
            "helpful_count": 5,
            "created": "2024-10-01T14:20:00Z",
            "_expanded": {"group": {"id": "teacher_smith", "name": "Dr. Smith", "type": "teacher"}}
        },
        {
            "id": "review_003",
            "group": "class_math301",
            "author": "user_0003",
            "rating": 5.0,
            "comment": "Best math class I've taken! The professor makes complex topics accessible and the problem sets are challenging but fair.",
            "rating_fields": {"clarity": 5, "usefulness": 5, "difficulty": 4},
            "is_anonymous": False,
            "helpful_count": 23,
            "created": "2024-10-10T09:15:00Z",
            "_expanded": {"group": {"id": "class_math301", "name": "Linear Algebra", "type": "class"}}
        }
    ]

    # Sample events
    sample_events = [
        {
            "id": "event_001",
            "title": "Fall Career Fair",
            "description": "Annual career fair featuring over 100 companies from tech, finance, and consulting industries. Bring your resume!",
            "location": "Student Union Ballroom",
            "date_time": "2024-11-15T10:00:00Z",
            "duration": 240,
            "host_group": "career_center",
            "attendees": ["user_0001", "user_0002", "user_0003"],
            "free_food": True,
            "tags": ["career", "networking", "professional"],
            "created": "2024-10-01T08:00:00Z",
            "_expanded": {"host_group": {"id": "career_center", "name": "Career Center", "type": "org"}}
        },
        {
            "id": "event_002",
            "title": "AI/ML Study Group",
            "description": "Weekly study session for machine learning enthusiasts. This week: Transformer architectures and attention mechanisms.",
            "location": "Library Room 204",
            "date_time": "2024-11-20T18:00:00Z",
            "duration": 120,
            "host_group": "cs_club",
            "attendees": ["user_0004", "user_0005"],
            "free_food": False,
            "tags": ["academic", "cs", "machine-learning", "study-group"],
            "created": "2024-11-10T12:00:00Z",
            "_expanded": {"host_group": {"id": "cs_club", "name": "Computer Science Club", "type": "org"}}
        }
    ]

    # Sample hangouts
    sample_hangouts = [
        {
            "id": "hangout_001",
            "title": "Coffee and Study Session",
            "description": "Looking for people to study with at the campus coffee shop. Working on CS201 homework.",
            "category": "academic",
            "tags": ["study", "coffee", "cs201"],
            "location": "Campus Coffee House",
            "date_time": "2024-11-18T14:00:00Z",
            "creator": "user_0001",
            "attendees": ["user_0001", "user_0006"],
            "everyone": True,
            "created": "2024-11-17T20:00:00Z"
        },
        {
            "id": "hangout_002",
            "title": "Pickup Basketball",
            "description": "Organizing a casual basketball game at the rec center. All skill levels welcome!",
            "category": "social",
            "tags": ["sports", "basketball", "fitness"],
            "location": "Recreation Center Gym",
            "date_time": "2024-11-19T16:00:00Z",
            "creator": "user_0007",
            "attendees": ["user_0007", "user_0008", "user_0009", "user_0010"],
            "everyone": True,
            "created": "2024-11-18T10:00:00Z"
        }
    ]

    # Sample posts
    sample_posts = [
        {
            "id": "post_001",
            "title": "Tips for surviving finals week",
            "text_content": "Hey everyone! Finals are coming up and I wanted to share some tips that helped me last semester:\n\n1. Start studying early - don't cram\n2. Take breaks every 45 minutes\n3. Stay hydrated and get enough sleep\n4. Form study groups for difficult subjects\n5. Use office hours - professors want to help!\n\nGood luck everyone!",
            "category": "academic",
            "tags": ["finals", "study-tips", "advice"],
            "author": "user_0011",
            "upvotes": 45,
            "downvotes": 2,
            "comment_count": 12,
            "is_anonymous": False,
            "created": "2024-11-15T08:30:00Z"
        },
        {
            "id": "post_002",
            "title": "Best study spots on campus?",
            "text_content": "I'm looking for quiet places to study that aren't the main library. Any recommendations? Preferably somewhere with good wifi and outlets.",
            "category": "social",
            "tags": ["study", "campus", "question"],
            "author": "user_0012",
            "upvotes": 18,
            "downvotes": 0,
            "comment_count": 25,
            "is_anonymous": False,
            "created": "2024-11-16T11:45:00Z"
        }
    ]

    # Sample classes
    sample_classes = [
        {
            "id": "class_cs201",
            "title": "Introduction to Computer Science",
            "description": "Foundational course covering programming fundamentals, data structures, and algorithmic thinking. Uses Python as the primary language.",
            "teachers": ["teacher_jones"],
            "majors": ["computer_science", "data_science"],
            "avg_rating": 4.2,
            "review_count": 156,
            "created": "2024-01-01T00:00:00Z",
            "_expanded": {"teachers": [{"id": "teacher_jones", "name": "Dr. Jones", "type": "teacher"}]}
        },
        {
            "id": "class_math301",
            "title": "Linear Algebra",
            "description": "Study of vector spaces, linear transformations, matrices, and systems of linear equations. Essential for machine learning and data science.",
            "teachers": ["teacher_chen"],
            "majors": ["mathematics", "computer_science", "physics"],
            "avg_rating": 4.5,
            "review_count": 89,
            "created": "2024-01-01T00:00:00Z",
            "_expanded": {"teachers": [{"id": "teacher_chen", "name": "Dr. Chen", "type": "teacher"}]}
        }
    ]

    # Sample teachers
    sample_teachers = [
        {
            "id": "teacher_jones",
            "name": "Dr. Sarah Jones",
            "bio": "Associate Professor of Computer Science with 10 years of teaching experience. Research focuses on programming language design and computer science education.",
            "research_description": "My research explores how novice programmers learn to code and how we can design better educational tools and curricula.",
            "classes": ["class_cs201", "class_cs301"],
            "majors": ["computer_science"],
            "avg_rating": 4.3,
            "review_count": 234,
            "created": "2024-01-01T00:00:00Z"
        },
        {
            "id": "teacher_chen",
            "name": "Dr. Michael Chen",
            "bio": "Professor of Mathematics specializing in linear algebra and numerical analysis. Known for making complex topics accessible.",
            "research_description": "Research in numerical linear algebra with applications to machine learning and scientific computing.",
            "classes": ["class_math301", "class_math401"],
            "majors": ["mathematics"],
            "avg_rating": 4.6,
            "review_count": 178,
            "created": "2024-01-01T00:00:00Z"
        }
    ]

    # Sample resources
    sample_resources = [
        {
            "id": "resource_001",
            "title": "CS201 Midterm Study Guide",
            "description": "Comprehensive study guide covering all topics from the first half of the semester. Includes practice problems with solutions.",
            "type": "study_guide",
            "tags": ["cs201", "midterm", "study-guide"],
            "group": "class_cs201",
            "author": "user_0013",
            "upvotes": 67,
            "downvotes": 3,
            "created": "2024-10-20T16:00:00Z",
            "_expanded": {"group": {"id": "class_cs201", "name": "Introduction to Computer Science", "type": "class"}}
        },
        {
            "id": "resource_002",
            "title": "Linear Algebra Cheat Sheet",
            "description": "One-page reference with all the key formulas and theorems for MATH301. Great for quick review before exams.",
            "type": "notes",
            "tags": ["math301", "cheat-sheet", "formulas"],
            "group": "class_math301",
            "author": "user_0014",
            "upvotes": 45,
            "downvotes": 1,
            "created": "2024-11-01T09:30:00Z",
            "_expanded": {"group": {"id": "class_math301", "name": "Linear Algebra", "type": "class"}}
        }
    ]

    # Sample groups
    sample_groups = [
        {
            "id": "class_cs201",
            "name": "Introduction to Computer Science",
            "type": "class",
            "description": "CS201 - Fall 2024",
            "tags": ["cs", "programming", "python"],
            "created": "2024-08-01T00:00:00Z"
        },
        {
            "id": "class_math301",
            "name": "Linear Algebra",
            "type": "class",
            "description": "MATH301 - Fall 2024",
            "tags": ["math", "linear-algebra"],
            "created": "2024-08-01T00:00:00Z"
        },
        {
            "id": "cs_club",
            "name": "Computer Science Club",
            "type": "org",
            "description": "A community for CS students to learn, collaborate, and grow together.",
            "tags": ["cs", "programming", "tech"],
            "created": "2024-01-01T00:00:00Z"
        }
    ]

    # Write all sample data
    data_files = {
        'reviews.json': sample_reviews,
        'events.json': sample_events,
        'hangouts.json': sample_hangouts,
        'posts.json': sample_posts,
        'classes.json': sample_classes,
        'teachers.json': sample_teachers,
        'resources.json': sample_resources,
        'groups.json': sample_groups
    }

    for filename, data in data_files.items():
        output_file = OUTPUT_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Created {output_file} with {len(data)} records")

    # Create .gitkeep files
    for subdir in ['processed', 'evaluation']:
        gitkeep = OUTPUT_DIR.parent / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()

    for subdir in ['metrics', 'figures', 'ablation']:
        gitkeep = OUTPUT_DIR.parent.parent / 'results' / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()

    print("\nSample data created successfully!")
    print("Note: This is placeholder data. Run with PocketBase for real data.")


def generate_synthetic_data():
    """
    Generate ~100 synthetic records for comprehensive testing.

    Creates realistic variety across all collections to properly
    test the RAG pipeline with semantic search and ranking.
    """
    print("Generating synthetic data (~100 records)...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)  # Reproducible data

    # =========================================================================
    # TEMPLATE DATA FOR GENERATION
    # =========================================================================

    # Departments and their classes
    DEPARTMENTS = {
        "Computer Science": {
            "prefix": "CS",
            "classes": [
                ("101", "Introduction to Programming", "Learn programming fundamentals using Python. Covers variables, loops, functions, and basic data structures."),
                ("201", "Data Structures", "Study of fundamental data structures including arrays, linked lists, trees, graphs, and hash tables."),
                ("301", "Algorithms", "Design and analysis of algorithms. Covers sorting, searching, graph algorithms, and dynamic programming."),
                ("350", "Database Systems", "Introduction to relational databases, SQL, normalization, and transaction processing."),
                ("401", "Machine Learning", "Fundamentals of machine learning including supervised learning, unsupervised learning, and neural networks."),
                ("450", "Software Engineering", "Principles of software development, agile methodologies, testing, and project management."),
                ("480", "Artificial Intelligence", "Introduction to AI concepts including search, knowledge representation, and reasoning."),
            ]
        },
        "Mathematics": {
            "prefix": "MATH",
            "classes": [
                ("151", "Calculus I", "Limits, derivatives, and integrals of single-variable functions."),
                ("152", "Calculus II", "Integration techniques, sequences, series, and parametric equations."),
                ("251", "Multivariable Calculus", "Partial derivatives, multiple integrals, and vector calculus."),
                ("301", "Linear Algebra", "Vector spaces, linear transformations, matrices, and eigenvalues."),
                ("310", "Probability Theory", "Probability spaces, random variables, and distributions."),
                ("350", "Differential Equations", "Ordinary differential equations and their applications."),
            ]
        },
        "Physics": {
            "prefix": "PHYS",
            "classes": [
                ("101", "General Physics I", "Mechanics, waves, and thermodynamics with calculus."),
                ("102", "General Physics II", "Electricity, magnetism, and optics."),
                ("301", "Quantum Mechanics", "Introduction to quantum theory and wave mechanics."),
                ("350", "Statistical Mechanics", "Statistical methods in physics and thermodynamics."),
            ]
        },
        "Biology": {
            "prefix": "BIOL",
            "classes": [
                ("101", "Introduction to Biology", "Cell biology, genetics, and evolution."),
                ("201", "Genetics", "Principles of heredity and molecular genetics."),
                ("301", "Biochemistry", "Chemical processes within living organisms."),
                ("350", "Neuroscience", "Structure and function of the nervous system."),
            ]
        },
        "English": {
            "prefix": "ENG",
            "classes": [
                ("101", "Composition I", "Academic writing and critical thinking."),
                ("201", "American Literature", "Survey of American literary traditions."),
                ("301", "Creative Writing", "Fiction, poetry, and creative nonfiction."),
            ]
        }
    }

    # Professor name templates
    FIRST_NAMES = ["Sarah", "Michael", "Jennifer", "David", "Emily", "James", "Maria", "Robert",
                   "Lisa", "William", "Amanda", "Christopher", "Jessica", "Thomas", "Ashley",
                   "Daniel", "Michelle", "Matthew", "Stephanie", "Andrew"]
    LAST_NAMES = ["Johnson", "Smith", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
                  "Martin", "Lee", "Thompson", "White", "Harris", "Chen", "Patel", "Kim", "Singh"]

    # Review comment templates (positive, neutral, negative)
    REVIEW_TEMPLATES = {
        "positive": [
            "Excellent professor! {professor} explains concepts clearly and is always available during office hours.",
            "One of the best classes I've taken. {professor} makes {subject} interesting and engaging.",
            "Highly recommend! The workload is manageable and {professor} is very supportive.",
            "{professor} is passionate about {subject} and it shows in every lecture.",
            "Great course! Learned so much about {subject}. {professor} provides excellent feedback.",
            "Challenging but rewarding. {professor} pushes you to think critically.",
            "Love this class! {professor} uses real-world examples that make {subject} click.",
        ],
        "neutral": [
            "Decent class overall. {professor} knows the material but lectures can be dry.",
            "Average experience. Some topics in {subject} were interesting, others less so.",
            "{professor} is knowledgeable but could be more engaging in lectures.",
            "The course covers important {subject} concepts. Exams are fair.",
            "Good fundamentals course. {professor} is helpful during office hours.",
            "Standard {subject} class. Nothing exceptional but gets the job done.",
        ],
        "negative": [
            "Struggled in this class. {professor}'s explanations of {subject} were confusing.",
            "Not a great experience. The workload was overwhelming and feedback was minimal.",
            "{professor} seems uninterested in teaching. {subject} deserves better.",
            "Lectures were hard to follow. Had to learn most of {subject} from the textbook.",
            "Would not recommend unless you have to take it. {professor} grades harshly.",
        ]
    }

    # Event templates
    EVENT_TEMPLATES = [
        ("Career Fair", "Annual career fair featuring companies from {industry}. Bring your resume!", ["career", "networking", "professional"]),
        ("{club} Workshop", "Hands-on workshop covering {topic}. All skill levels welcome!", ["workshop", "learning", "hands-on"]),
        ("Guest Speaker: {speaker}", "Join us for a talk on {topic} by industry expert {speaker}.", ["speaker", "lecture", "professional"]),
        ("{subject} Study Session", "Group study session for {subject}. We'll review key concepts and work through practice problems.", ["study", "academic", "group"]),
        ("Networking Night", "Connect with alumni and professionals in {industry}. Food and drinks provided!", ["networking", "alumni", "social"]),
        ("{club} Social", "Casual hangout for {club} members. Games, snacks, and good company!", ["social", "club", "fun"]),
        ("Hackathon: {theme}", "24-hour coding competition. Build something awesome with {theme}!", ["hackathon", "coding", "competition"]),
        ("Research Symposium", "Undergraduate research presentations in {subject}.", ["research", "academic", "presentation"]),
    ]

    # Post content templates
    POST_TEMPLATES = [
        ("Tips for {subject}", "Here are some tips that helped me succeed in {subject}:\n\n1. {tip1}\n2. {tip2}\n3. {tip3}\n\nGood luck!", ["tips", "advice", "academic"]),
        ("Looking for study partners", "Anyone taking {class_name} this semester? Looking to form a study group.", ["study-group", "looking-for", "academic"]),
        ("Best resources for {subject}?", "What are the best resources you've found for learning {subject}? Books, videos, websites?", ["resources", "question", "help"]),
        ("Internship advice needed", "I'm applying for {industry} internships. Any tips on standing out?", ["internship", "career", "advice"]),
        ("{subject} project ideas?", "Need ideas for my {subject} final project. What have you all done?", ["project", "ideas", "help"]),
        ("Study spots on campus", "Where are your favorite places to study? Looking for somewhere quiet with good wifi.", ["study", "campus", "question"]),
        ("Professor recommendations for {subject}", "Which professors would you recommend for {subject} courses?", ["professor", "recommendation", "question"]),
    ]

    # Hangout templates
    HANGOUT_TEMPLATES = [
        ("Coffee and Study", "Looking for people to study with at the coffee shop. Working on {class_name}.", "academic", ["study", "coffee", "group"]),
        ("Pickup Basketball", "Organizing a casual game at the rec center. All skill levels!", "social", ["sports", "basketball", "fitness"]),
        ("Board Game Night", "Hosting board games at my place. Bring snacks!", "social", ["games", "fun", "social"]),
        ("Hiking Trip", "Planning a hike this weekend. Meeting at {location}.", "social", ["outdoors", "hiking", "nature"]),
        ("Movie Marathon", "Watching {genre} movies. Pizza provided!", "social", ["movies", "entertainment", "fun"]),
        ("Study Group: {class_name}", "Weekly study session for {class_name}. Library room 204.", "academic", ["study", "group", "academic"]),
        ("Gym Buddies", "Looking for workout partners. Usually go in the {time}.", "social", ["fitness", "gym", "health"]),
        ("Food Adventure", "Trying new restaurant: {restaurant}. Who's in?", "social", ["food", "dining", "adventure"]),
    ]

    # Resource templates
    RESOURCE_TYPES = ["study_guide", "notes", "cheat_sheet", "practice_problems", "lecture_notes", "summary"]

    # Locations
    LOCATIONS = ["Student Union", "Library", "Engineering Building", "Science Center", "Arts Building",
                 "Recreation Center", "Coffee House", "Quad", "Room 101", "Room 204", "Auditorium"]

    # =========================================================================
    # GENERATE DATA
    # =========================================================================

    generated_teachers = []
    generated_classes = []
    generated_groups = []
    generated_reviews = []
    generated_events = []
    generated_hangouts = []
    generated_posts = []
    generated_resources = []

    # Track IDs
    teacher_id = 1
    class_id = 1
    group_id = 1
    review_id = 1
    event_id = 1
    hangout_id = 1
    post_id = 1
    resource_id = 1

    # Generate teachers and classes per department
    for dept_name, dept_info in DEPARTMENTS.items():
        prefix = dept_info["prefix"]

        # Generate 2-3 teachers per department
        num_teachers = random.randint(2, 3)
        dept_teachers = []

        for _ in range(num_teachers):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)

            teacher = {
                "id": f"teacher_{teacher_id:03d}",
                "name": f"Dr. {first} {last}",
                "bio": f"Professor of {dept_name} with expertise in various areas of the field.",
                "research_description": f"My research focuses on advancing knowledge in {dept_name.lower()} through innovative approaches.",
                "classes": [],
                "majors": [dept_name.lower().replace(" ", "_")],
                "avg_rating": round(random.uniform(3.0, 5.0), 1),
                "review_count": random.randint(20, 200),
                "created": "2024-01-01T00:00:00Z"
            }
            generated_teachers.append(teacher)
            dept_teachers.append(teacher)
            teacher_id += 1

        # Generate classes for department
        for course_num, course_name, course_desc in dept_info["classes"]:
            assigned_teacher = random.choice(dept_teachers)

            class_record = {
                "id": f"class_{class_id:03d}",
                "title": f"{prefix}{course_num}: {course_name}",
                "description": course_desc,
                "teachers": [assigned_teacher["id"]],
                "majors": [dept_name.lower().replace(" ", "_")],
                "avg_rating": round(random.uniform(3.5, 4.8), 1),
                "review_count": random.randint(30, 150),
                "created": "2024-01-01T00:00:00Z",
                "_expanded": {"teachers": [{"id": assigned_teacher["id"], "name": assigned_teacher["name"], "type": "teacher"}]}
            }
            generated_classes.append(class_record)

            # Add class to teacher
            assigned_teacher["classes"].append(class_record["id"])

            # Create group for this class
            group_record = {
                "id": f"group_{group_id:03d}",
                "name": f"{prefix}{course_num}: {course_name}",
                "type": "class",
                "description": f"{prefix}{course_num} - Fall 2024",
                "tags": [prefix.lower(), course_name.lower().split()[0]],
                "created": "2024-08-01T00:00:00Z"
            }
            generated_groups.append(group_record)
            group_id += 1

            # Generate 1-3 reviews per class
            num_reviews = random.randint(1, 3)
            for _ in range(num_reviews):
                rating = random.uniform(2.0, 5.0)
                if rating >= 4.0:
                    template = random.choice(REVIEW_TEMPLATES["positive"])
                elif rating >= 3.0:
                    template = random.choice(REVIEW_TEMPLATES["neutral"])
                else:
                    template = random.choice(REVIEW_TEMPLATES["negative"])

                comment = template.format(
                    professor=assigned_teacher["name"],
                    subject=course_name.lower()
                )

                days_ago = random.randint(1, 180)
                created_date = datetime.now() - timedelta(days=days_ago)

                review = {
                    "id": f"review_{review_id:03d}",
                    "group": class_record["id"],
                    "author": f"user_{random.randint(1, 50):04d}",
                    "rating": round(rating, 1),
                    "comment": comment,
                    "rating_fields": {
                        "clarity": random.randint(2, 5),
                        "helpfulness": random.randint(2, 5),
                        "workload": random.randint(1, 5)
                    },
                    "is_anonymous": random.choice([True, False]),
                    "helpful_count": random.randint(0, 30),
                    "created": created_date.isoformat() + "Z",
                    "_expanded": {"group": {"id": class_record["id"], "name": class_record["title"], "type": "class"}}
                }
                generated_reviews.append(review)
                review_id += 1

            # Generate 0-2 resources per class
            num_resources = random.randint(0, 2)
            for _ in range(num_resources):
                res_type = random.choice(RESOURCE_TYPES)
                res_title = f"{prefix}{course_num} {res_type.replace('_', ' ').title()}"

                resource = {
                    "id": f"resource_{resource_id:03d}",
                    "title": res_title,
                    "description": f"Helpful {res_type.replace('_', ' ')} for {course_name}. Covers key concepts and formulas.",
                    "type": res_type,
                    "tags": [prefix.lower(), course_num, res_type],
                    "group": class_record["id"],
                    "author": f"user_{random.randint(1, 50):04d}",
                    "upvotes": random.randint(5, 80),
                    "downvotes": random.randint(0, 5),
                    "created": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat() + "Z",
                    "_expanded": {"group": {"id": class_record["id"], "name": class_record["title"], "type": "class"}}
                }
                generated_resources.append(resource)
                resource_id += 1

            class_id += 1

    # Generate organizations
    ORGS = [
        ("Computer Science Club", "org", "A community for CS students to learn, collaborate, and grow together.", ["cs", "programming", "tech"]),
        ("Math Club", "org", "For students passionate about mathematics and problem-solving.", ["math", "problem-solving"]),
        ("Engineering Society", "org", "Supporting engineering students through networking and professional development.", ["engineering", "professional"]),
        ("Data Science Club", "org", "Exploring data science, machine learning, and analytics together.", ["data", "ml", "analytics"]),
        ("Women in STEM", "org", "Empowering women in science, technology, engineering, and mathematics.", ["diversity", "stem", "support"]),
        ("Career Center", "org", "Helping students prepare for their professional futures.", ["career", "jobs", "professional"]),
        ("Student Government", "org", "Representing student interests and organizing campus events.", ["government", "student-life"]),
        ("Robotics Club", "org", "Building robots and competing in robotics competitions.", ["robotics", "engineering", "competition"]),
    ]

    for org_name, org_type, org_desc, org_tags in ORGS:
        org = {
            "id": f"group_{group_id:03d}",
            "name": org_name,
            "type": org_type,
            "description": org_desc,
            "tags": org_tags,
            "created": "2024-01-01T00:00:00Z"
        }
        generated_groups.append(org)
        group_id += 1

    # Generate events (12-15 events)
    industries = ["tech", "finance", "consulting", "healthcare", "startups"]
    topics = ["machine learning", "web development", "data analysis", "cloud computing", "interview prep", "resume writing"]
    themes = ["sustainability", "social good", "gaming", "health tech", "education"]
    clubs = ["CS Club", "Data Science Club", "Engineering Society", "Math Club"]
    speakers = ["Alex Rivera", "Jordan Chen", "Sam Williams", "Morgan Taylor", "Casey Adams"]

    for i in range(random.randint(12, 15)):
        template = random.choice(EVENT_TEMPLATES)
        title_template, desc_template, base_tags = template

        title = title_template.format(
            club=random.choice(clubs),
            speaker=random.choice(speakers),
            subject=random.choice(list(DEPARTMENTS.keys())),
            industry=random.choice(industries),
            topic=random.choice(topics),
            theme=random.choice(themes)
        )

        description = desc_template.format(
            club=random.choice(clubs),
            speaker=random.choice(speakers),
            subject=random.choice(list(DEPARTMENTS.keys())),
            industry=random.choice(industries),
            topic=random.choice(topics),
            theme=random.choice(themes)
        )

        # Future date for events
        days_ahead = random.randint(1, 60)
        event_date = datetime.now() + timedelta(days=days_ahead)

        host_org = random.choice([g for g in generated_groups if g["type"] == "org"])

        event = {
            "id": f"event_{event_id:03d}",
            "title": title,
            "description": description,
            "location": random.choice(LOCATIONS),
            "date_time": event_date.isoformat() + "Z",
            "duration": random.choice([60, 90, 120, 180, 240]),
            "host_group": host_org["id"],
            "attendees": [f"user_{random.randint(1, 50):04d}" for _ in range(random.randint(3, 20))],
            "free_food": random.choice([True, False, False]),  # 33% chance of free food
            "tags": base_tags + [random.choice(["fall-2024", "weekly", "monthly"])],
            "created": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z",
            "_expanded": {"host_group": {"id": host_org["id"], "name": host_org["name"], "type": "org"}}
        }
        generated_events.append(event)
        event_id += 1

    # Generate hangouts (10-12 hangouts)
    genres = ["sci-fi", "comedy", "action", "horror"]
    restaurants = ["Thai Palace", "Burger Barn", "Sushi Express", "Taco Haven", "Pizza Planet"]
    times = ["morning", "afternoon", "evening"]

    for i in range(random.randint(10, 12)):
        template = random.choice(HANGOUT_TEMPLATES)
        title_template, desc_template, category, base_tags = template

        random_class = random.choice(generated_classes)

        title = title_template.format(
            class_name=random_class["title"].split(":")[0],
            genre=random.choice(genres),
            restaurant=random.choice(restaurants),
            location="Campus Trailhead"
        )

        description = desc_template.format(
            class_name=random_class["title"].split(":")[0],
            genre=random.choice(genres),
            restaurant=random.choice(restaurants),
            location="Campus Trailhead",
            time=random.choice(times)
        )

        days_ahead = random.randint(0, 14)
        hangout_date = datetime.now() + timedelta(days=days_ahead)

        hangout = {
            "id": f"hangout_{hangout_id:03d}",
            "title": title,
            "description": description,
            "category": category,
            "tags": base_tags,
            "location": random.choice(LOCATIONS),
            "date_time": hangout_date.isoformat() + "Z",
            "creator": f"user_{random.randint(1, 50):04d}",
            "attendees": [f"user_{random.randint(1, 50):04d}" for _ in range(random.randint(1, 8))],
            "everyone": random.choice([True, True, False]),  # 66% open to everyone
            "created": (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat() + "Z"
        }
        generated_hangouts.append(hangout)
        hangout_id += 1

    # Generate posts (12-15 posts)
    tips = [
        "Start assignments early",
        "Form study groups",
        "Use office hours",
        "Practice with past exams",
        "Take good notes",
        "Review before each class",
        "Get enough sleep",
        "Stay organized"
    ]

    for i in range(random.randint(12, 15)):
        template = random.choice(POST_TEMPLATES)
        title_template, content_template, base_tags = template

        random_class = random.choice(generated_classes)
        subject = random.choice(list(DEPARTMENTS.keys()))

        title = title_template.format(
            subject=subject,
            class_name=random_class["title"],
            industry=random.choice(industries)
        )

        selected_tips = random.sample(tips, 3)
        content = content_template.format(
            subject=subject,
            class_name=random_class["title"],
            industry=random.choice(industries),
            tip1=selected_tips[0],
            tip2=selected_tips[1],
            tip3=selected_tips[2]
        )

        days_ago = random.randint(0, 60)
        post_date = datetime.now() - timedelta(days=days_ago)

        post = {
            "id": f"post_{post_id:03d}",
            "title": title,
            "text_content": content,
            "category": random.choice(["academic", "social", "career", "general"]),
            "tags": base_tags + [subject.lower().replace(" ", "-")],
            "author": f"user_{random.randint(1, 50):04d}",
            "upvotes": random.randint(0, 60),
            "downvotes": random.randint(0, 5),
            "comment_count": random.randint(0, 30),
            "is_anonymous": random.choice([True, False, False]),  # 33% anonymous
            "created": post_date.isoformat() + "Z"
        }
        generated_posts.append(post)
        post_id += 1

    # =========================================================================
    # WRITE DATA FILES
    # =========================================================================

    data_files = {
        'reviews.json': generated_reviews,
        'events.json': generated_events,
        'hangouts.json': generated_hangouts,
        'posts.json': generated_posts,
        'classes.json': generated_classes,
        'teachers.json': generated_teachers,
        'resources.json': generated_resources,
        'groups.json': generated_groups
    }

    total_records = 0
    for filename, data in data_files.items():
        output_file = OUTPUT_DIR / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Created {filename}: {len(data)} records")
        total_records += len(data)

    # Create .gitkeep files
    for subdir in ['processed', 'evaluation']:
        gitkeep = OUTPUT_DIR.parent / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()

    for subdir in ['metrics', 'figures', 'ablation']:
        gitkeep = OUTPUT_DIR.parent.parent / 'results' / subdir / '.gitkeep'
        gitkeep.parent.mkdir(parents=True, exist_ok=True)
        gitkeep.touch()

    print(f"\n{'='*50}")
    print(f"SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Total records: {total_records}")
    print(f"  - Teachers: {len(generated_teachers)}")
    print(f"  - Classes: {len(generated_classes)}")
    print(f"  - Groups: {len(generated_groups)}")
    print(f"  - Reviews: {len(generated_reviews)}")
    print(f"  - Events: {len(generated_events)}")
    print(f"  - Hangouts: {len(generated_hangouts)}")
    print(f"  - Posts: {len(generated_posts)}")
    print(f"  - Resources: {len(generated_resources)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")


async def main():
    """Main entry point."""
    print("=" * 50)
    print("TRIBLY DATA EXPORT TOOL")
    print("=" * 50)

    # Check for command line flags
    if '--help' in sys.argv or '-h' in sys.argv:
        print("\nUsage: python scripts/export_data.py [OPTIONS]")
        print("\nOptions:")
        print("  --sample    Create minimal sample data (18 records)")
        print("  --generate  Generate ~100 synthetic records for testing")
        print("  --help, -h  Show this help message")
        print("\nWith no options, attempts to export from PocketBase,")
        print("falling back to sample data if unavailable.")
        return

    if '--generate' in sys.argv:
        generate_synthetic_data()
        return

    if '--sample' in sys.argv:
        create_sample_data()
        return

    # Check if httpx is available for PocketBase export
    if not HTTPX_AVAILABLE:
        print("\nhttpx not installed. Creating sample data instead.")
        print("Install dependencies with: pip install -r requirements.txt")
        print("Or use --generate for comprehensive test data.")
        create_sample_data()
        return

    # Try to export from PocketBase
    try:
        await export_all_collections()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nFalling back to sample data...")
        create_sample_data()


if __name__ == '__main__':
    asyncio.run(main())
