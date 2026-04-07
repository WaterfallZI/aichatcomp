from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import hashlib, os

db = SQLAlchemy()

PLANS = {
    'free':  {'credits': 50,   'bonus': 300, 'label': 'Free'},
    'pro':   {'credits': 1000, 'bonus': 0,   'label': 'Pro'},
    'max':   {'credits': 5000, 'bonus': 0,   'label': 'Max'},
    'ultra': {'credits': -1,   'bonus': 0,   'label': 'Ultra'},
}

class User(db.Model):
    """Only original columns — new ones managed via raw SQL to avoid migration issues"""
    __tablename__ = 'users'

    id               = db.Column(db.Integer, primary_key=True)
    email            = db.Column(db.String(255), unique=True, nullable=False)
    username         = db.Column(db.String(100), nullable=False)
    password_hash    = db.Column(db.String(128), nullable=True)
    nomchat_id       = db.Column(db.Integer, nullable=True)
    nomchat_username = db.Column(db.String(100), nullable=True)
    nomchat_avatar   = db.Column(db.String(50), nullable=True)
    is_banned        = db.Column(db.Boolean, default=False)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    last_login       = db.Column(db.DateTime)

    @staticmethod
    def hash_password(password: str) -> str:
        salt = os.environ.get('PASSWORD_SALT', 'aichat_salt_2026')
        return hashlib.sha256(f'{salt}{password}'.encode()).hexdigest()

    def verify_password(self, password: str) -> bool:
        return self.password_hash == self.hash_password(password)


class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'

    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_key = db.Column(db.String(64), unique=True, nullable=False)
    operator_on = db.Column(db.Boolean, default=False)
    operator_id = db.Column(db.Integer, nullable=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user        = db.relationship('User', backref='chat_sessions', foreign_keys=[user_id])

    def to_dict(self, include_messages=False):
        d = {
            'id': self.id, 'user_id': self.user_id,
            'session_key': self.session_key,
            'operator_on': self.operator_on, 'operator_id': self.operator_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'user': {'id': self.user.id, 'username': self.user.username,
                     'email': self.user.email, 'avatar': self.user.nomchat_avatar or '👤'
                     } if self.user else None,
        }
        if include_messages:
            d['messages'] = [m.to_dict() for m in
                             ChatMessage.query.filter_by(session_id=self.id)
                             .order_by(ChatMessage.created_at).all()]
        return d


class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False)
    role       = db.Column(db.String(20), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {'id': self.id, 'session_id': self.session_id,
                'role': self.role, 'content': self.content,
                'created_at': self.created_at.isoformat()}
