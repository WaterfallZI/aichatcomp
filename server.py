"""
AI Chat Company вЂ” Optimized Flask Server
Features: Auth, Groq AI, Sessions, Credits, Operator, Image Gen, Voice, Preview
"""
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, secrets, requests, traceback, json, time

app = Flask(__name__, static_folder='.')

# в”Ђв”Ђ Config в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
app.secret_key = os.environ.get('SECRET_KEY', 'aichat-dev-secret-2026')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('RAILWAY_ENVIRONMENT') == 'production',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

_db_url = os.environ.get('DATABASE_URL', 'sqlite:///aichat.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
_db_url = re.sub(r'[?&]channel_binding=[^&]*', '', _db_url)
_db_url = re.sub(r'\?$|&$', '', _db_url)

app.config.update(
    SQLALCHEMY_DATABASE_URI=_db_url,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
    }
)

CORS(app, supports_credentials=True,
     origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))
db = SQLAlchemy(app)

# в”Ђв”Ђ Constants в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
GROQ_URL    = 'https://openrouter.ai/api/v1/chat/completions'
NOMCHAT_URL = os.environ.get('NOMCHAT_URL', 'https://nomchat-id.up.railway.app')
OPERATOR_EMAIL = os.environ.get('OPERATOR_EMAIL', 'ai@com.ru')

PLANS = {
    'free':  {'credits': 50,   'bonus': 300,  'label': 'Free'},
    'pro':   {'credits': 1000, 'bonus': 0,    'label': 'Pro'},
    'max':   {'credits': 5000, 'bonus': 0,    'label': 'Max'},
    'ultra': {'credits': -1,   'bonus': 0,    'label': 'Ultra'},
}

# в”Ђв”Ђ Models в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
class User(db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username       = db.Column(db.String(80), nullable=False)
    password_hash  = db.Column(db.String(256))
    nomchat_id     = db.Column(db.String(64), index=True)
    nomchat_username = db.Column(db.String(80))
    nomchat_avatar = db.Column(db.String(10), default='рџ’¬')
    credits        = db.Column(db.Integer, default=50)
    bonus_credits  = db.Column(db.Integer, default=300)
    plan           = db.Column(db.String(20), default='free')
    is_banned      = db.Column(db.Boolean, default=False)
    is_operator    = db.Column(db.Boolean, default=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    last_login     = db.Column(db.DateTime)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return bool(self.password_hash and check_password_hash(self.password_hash, pw))

    def to_dict(self):
        total = -1 if self.plan == 'ultra' else (self.credits or 0) + (self.bonus_credits or 0)
        return {
            'id': self.id, 'email': self.email, 'username': self.username,
            'credits': self.credits or 0, 'bonus_credits': self.bonus_credits or 0,
            'total_credits': total, 'plan': self.plan or 'free',
            'nomchat_id': self.nomchat_id, 'nomchat_username': self.nomchat_username,
            'nomchat_avatar': self.nomchat_avatar or 'рџ’¬',
            'has_password': bool(self.password_hash),
            'is_operator': self.is_operator or False,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

    def deduct(self, amount=1):
        if self.plan == 'ultra': return
        if (self.bonus_credits or 0) >= amount:
            self.bonus_credits -= amount
        elif (self.credits or 0) >= amount:
            self.credits -= amount
        else:
            total = (self.bonus_credits or 0) + (self.credits or 0)
            self.bonus_credits = 0
            self.credits = max(0, total - amount)
        db.session.commit()

    def has_credits(self, amount=1):
        if self.plan == 'ultra': return True
        return ((self.credits or 0) + (self.bonus_credits or 0)) >= amount

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    session_key = db.Column(db.String(64), unique=True, default=lambda: secrets.token_urlsafe(24))
    operator_on = db.Column(db.Boolean, default=False)
    operator_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at  = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages    = db.relationship('ChatMessage', backref='session', lazy='dynamic',
                                  cascade='all, delete-orphan')

    def to_dict(self, include_messages=False):
        d = {
            'id': self.id, 'session_key': self.session_key,
            'user_id': self.user_id,
            'operator_on': self.operator_on,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_messages:
            d['messages'] = [m.to_dict() for m in self.messages.order_by(ChatMessage.created_at)]
        return d

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False, index=True)
    role       = db.Column(db.String(20), nullable=False)  # user | ai | operator
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id, 'role': self.role, 'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

# в”Ђв”Ђ DB Init в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
with app.app_context():
    db.create_all()

# в”Ђв”Ђ Rate limiting (in-memory) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
_rate_store: dict = {}

def rate_limit(key: str, max_calls: int = 10, window: int = 60) -> bool:
    now = time.time()
    calls = [t for t in _rate_store.get(key, []) if now - t < window]
    if len(calls) >= max_calls:
        return False
    calls.append(now)
    _rate_store[key] = calls
    return True

# в”Ђв”Ђ Auth decorators в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user:
            session.clear()
            return jsonify({'error': 'Unauthorized'}), 401
        if user.is_banned:
            return jsonify({'error': 'banned'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

def operator_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user or (user.email != OPERATOR_EMAIL and not user.is_operator):
            return jsonify({'error': 'Forbidden'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

# в”Ђв”Ђ Static files в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('.', filename)
    except Exception:
        try:
            return send_from_directory('.', '404.html'), 404
        except Exception:
            return jsonify({'error': 'Not found'}), 404

@app.errorhandler(404)
def not_found(e):
    try:
        return send_from_directory('.', '404.html'), 404
    except Exception:
        return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f'500: {e}\n{traceback.format_exc()}')
    return jsonify({'error': 'Internal server error'}), 500

# в”Ђв”Ђ Auth: Register в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/auth/register', methods=['POST'])
def register():
    return jsonify({'error': 'Email registration disabled. Use Nomchat ID.'}), 403

@app.route('/api/auth/login', methods=['POST'])
def login():
    return jsonify({'error': 'Email login disabled. Use Nomchat ID.'}), 403

@app.route('/api/auth/nomchat', methods=['POST'])
def nomchat_auth():
    d = request.get_json(silent=True) or {}
    token = d.get('token', '').strip()
    if not token:
        return jsonify({'error': 'Token required'}), 400

    try:
        res = requests.post(
            f'{NOMCHAT_URL}/api/auth/token/verify',
            json={'token': token, 'app_id': 'ai-chat-pro'},
            timeout=8
        )
        nc = res.json()
    except Exception as e:
        return jsonify({'error': f'Nomchat unreachable: {e}'}), 503

    if not res.ok or not nc.get('success'):
        return jsonify({'error': nc.get('error', 'Invalid token')}), 401

    nc_user = nc['user']
    email   = nc_user.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'No email from Nomchat'}), 400

    try:
        user = User.query.filter_by(email=email).first()
        is_new = user is None
        if not user:
            user = User(
                email=email,
                username=nc_user.get('username', email.split('@')[0]),
                nomchat_id=str(nc_user.get('id', '')),
                nomchat_username=nc_user.get('username', ''),
                nomchat_avatar=nc_user.get('avatar', 'рџ’¬'),
            )
            db.session.add(user)
        else:
            user.nomchat_id       = str(nc_user.get('id', ''))
            user.nomchat_username = nc_user.get('username', user.nomchat_username)
            user.nomchat_avatar   = nc_user.get('avatar', user.nomchat_avatar or 'рџ’¬')

        if user.is_banned:
            return jsonify({'error': 'Account banned'}), 403

        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': is_new})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Nomchat auth error: {e}')
        return jsonify({'error': 'Authentication failed'}), 500

@app.route('/api/auth/me')
@login_required
def me(user):
    return jsonify(user.to_dict())

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

# в”Ђв”Ђ Config: Groq Key в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/config/groq-key')
def get_groq_key():
    key = os.environ.get('OPENROUTER_KEY', '')
    if not key:
        return jsonify({'error': 'Groq not configured'}), 503
    return jsonify({'key': key})

# в”Ђв”Ђ Chat: Groq proxy with streaming в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/chat', methods=['POST'])
@login_required
def chat(user):
    if not user.has_credits():
        return jsonify({'error': 'no_credits', 'message': 'No credits remaining.'}), 402

    groq_key = os.environ.get('OPENROUTER_KEY', '')
    if not groq_key:
        return jsonify({'error': 'AI service not configured'}), 503

    d = request.get_json(silent=True) or {}
    messages = d.get('messages', [])
    model    = d.get('model', 'llama-3.3-70b-versatile')
    stream   = d.get('stream', False)

    # Validate model
    ALLOWED_MODELS = {
        'meta-llama/llama-3.3-70b-instruct',
        'meta-llama/llama-3.1-8b-instruct',
        'mistralai/mixtral-8x7b-instruct',
        'google/gemma-2-9b-it',
        'meta-llama/llama-4-scout',
        'anthropic/claude-3-haiku',
        'openai/gpt-4o-mini',
        'deepseek/deepseek-chat',
    }
    if model not in ALLOWED_MODELS:
        model = 'meta-llama/llama-3.3-70b-instruct'

    # Filter messages вЂ” only string content for Groq
    clean_msgs = []
    for m in messages:
        role    = m.get('role', 'user')
        content = m.get('content', '')
        if role not in ('system', 'user', 'assistant'):
            continue
        if isinstance(content, str) and content.strip():
            clean_msgs.append({'role': role, 'content': content.strip()})
        elif isinstance(content, list):
            # multimodal вЂ” keep as-is for vision models
            clean_msgs.append({'role': role, 'content': content})

    if not clean_msgs:
        return jsonify({'error': 'No messages provided'}), 400

    payload = {
        'model': model,
        'messages': clean_msgs,
        'max_tokens': 2048,
        'temperature': 0.7,
        'top_p': 0.9,
        'stream': stream,
    }

    headers = {
        'Authorization': f'Bearer {groq_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://aichatcompany.up.railway.app',
        'X-Title': 'AI Chat Company',
    }

    try:
        if stream:
            # Streaming response
            def generate():
                with requests.post(GROQ_URL, headers=headers, json=payload,
                                   stream=True, timeout=60) as r:
                    if not r.ok:
                        err = r.json().get('error', {})
                        yield f"data: {json.dumps({'error': err.get('message', 'Groq error')})}\n\n"
                        return
                    for line in r.iter_lines():
                        if line:
                            yield line.decode('utf-8') + '\n\n'
                user.deduct()

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}
            )
        else:
            # Non-streaming
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if not r.ok:
                err_data = r.json()
                err_msg  = err_data.get('error', {}).get('message', f'Groq error {r.status_code}')
                app.logger.warning(f'Groq error: {r.status_code} {err_msg}')
                return jsonify({'error': err_msg}), r.status_code

            result = r.json()
            user.deduct()
            total = -1 if user.plan == 'ultra' else (user.credits or 0) + (user.bonus_credits or 0)
            result['credits_remaining'] = total
            return jsonify(result)

    except requests.Timeout:
        return jsonify({'error': 'AI request timed out. Please try again.'}), 504
    except Exception as e:
        app.logger.error(f'Chat error: {e}')
        return jsonify({'error': str(e)}), 500

# в”Ђв”Ђ User: Update / Delete в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/user/update', methods=['POST'])
@login_required
def update_user(user):
    d = request.get_json(silent=True) or {}
    if 'username' in d:
        name = d['username'].strip()
        if 2 <= len(name) <= 50:
            user.username = name
    if 'password' in d and len(d['password']) >= 6:
        user.set_password(d['password'])
    db.session.commit()
    return jsonify(user.to_dict())

@app.route('/api/user/delete', methods=['DELETE'])
@login_required
def delete_user(user):
    db.session.delete(user)
    db.session.commit()
    session.clear()
    return jsonify({'success': True})

# в”Ђв”Ђ Plans в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/plans')
def get_plans():
    return jsonify(PLANS)

# в”Ђв”Ђ Chat Sessions (operator) в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/session/get', methods=['POST'])
@login_required
def get_session(user):
    sess = ChatSession.query.filter_by(user_id=user.id)\
               .order_by(ChatSession.created_at.desc()).first()
    if not sess:
        sess = ChatSession(user_id=user.id)
        db.session.add(sess)
        db.session.commit()
    return jsonify(sess.to_dict(include_messages=True))

@app.route('/api/session/message', methods=['POST'])
@login_required
def save_message(user):
    d       = request.get_json(silent=True) or {}
    content = d.get('content', '').strip()
    role    = d.get('role', 'user')
    if not content:
        return jsonify({'error': 'Empty message'}), 400
    if role not in ('user', 'ai'):
        return jsonify({'error': 'Invalid role'}), 400

    sess = ChatSession.query.filter_by(user_id=user.id)\
               .order_by(ChatSession.created_at.desc()).first()
    if not sess:
        sess = ChatSession(user_id=user.id)
        db.session.add(sess)
        db.session.flush()

    msg = ChatMessage(session_id=sess.id, role=role, content=content)
    db.session.add(msg)
    sess.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'ok': True, 'operator_on': sess.operator_on, 'message': msg.to_dict()})

@app.route('/api/session/poll')
@login_required
def poll_session(user):
    since_id = int(request.args.get('since', 0))
    sess = ChatSession.query.filter_by(user_id=user.id)\
               .order_by(ChatSession.created_at.desc()).first()
    if not sess:
        return jsonify({'operator_on': False, 'messages': []})
    msgs = ChatMessage.query.filter(
        ChatMessage.session_id == sess.id,
        ChatMessage.id > since_id
    ).order_by(ChatMessage.created_at).all()
    return jsonify({
        'operator_on': sess.operator_on,
        'messages': [m.to_dict() for m in msgs],
        'session_id': sess.id,
    })

# в”Ђв”Ђ Operator в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/operator/sessions')
@operator_required
def operator_sessions(user):
    sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).limit(100).all()
    result = []
    for s in sessions:
        d = s.to_dict()
        u = db.session.get(User, s.user_id)
        d['user'] = {'id': u.id, 'username': u.username, 'email': u.email} if u else None
        d['messages'] = [m.to_dict() for m in s.messages.order_by(ChatMessage.created_at).limit(1).all()]
        result.append(d)
    return jsonify(result)

@app.route('/api/operator/session/<int:sid>')
@operator_required
def operator_get_session(user, sid):
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    d = sess.to_dict(include_messages=True)
    u = db.session.get(User, sess.user_id)
    d['user_id'] = sess.user_id
    d['user'] = u.to_dict() if u else None
    return jsonify(d)

@app.route('/api/operator/takeover', methods=['POST'])
@operator_required
def operator_takeover(user):
    sid  = (request.get_json(silent=True) or {}).get('session_id')
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    sess.operator_on = True
    sess.operator_id = user.id
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/operator/release', methods=['POST'])
@operator_required
def operator_release(user):
    sid  = (request.get_json(silent=True) or {}).get('session_id')
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    sess.operator_on = False
    sess.operator_id = None
    db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/operator/send', methods=['POST'])
@operator_required
def operator_send(user):
    d       = request.get_json(silent=True) or {}
    sid     = d.get('session_id')
    content = d.get('content', '').strip()
    if not content: return jsonify({'error': 'Empty'}), 400
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    msg = ChatMessage(session_id=sess.id, role='operator', content=content)
    db.session.add(msg)
    sess.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'ok': True, 'message': msg.to_dict()})

@app.route('/api/operator/poll/<int:sid>')
@operator_required
def operator_poll(user, sid):
    since_id = int(request.args.get('since', 0))
    msgs = ChatMessage.query.filter(
        ChatMessage.session_id == sid,
        ChatMessage.id > since_id
    ).order_by(ChatMessage.created_at).all()
    return jsonify({'messages': [m.to_dict() for m in msgs]})

# в”Ђв”Ђ Image Generation в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/image/generate', methods=['POST'])
@login_required
def generate_image(user):
    if not user.has_credits(5):
        return jsonify({'error': 'Need at least 5 credits for image generation'}), 402

    d      = request.get_json(silent=True) or {}
    prompt = d.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt required'}), 400

    try:
        resp = requests.post(
            'https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0',
            headers={'Content-Type': 'application/json'},
            json={'inputs': prompt},
            timeout=45
        )
        if resp.ok and resp.headers.get('content-type', '').startswith('image/'):
            import base64
            img_b64 = base64.b64encode(resp.content).decode('utf-8')
            user.deduct(5)
            total = -1 if user.plan == 'ultra' else (user.credits or 0) + (user.bonus_credits or 0)
            return jsonify({
                'success': True,
                'image': f'data:image/png;base64,{img_b64}',
                'credits_remaining': total,
            })
        return jsonify({'error': 'Image generation service unavailable. Try again later.'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# в”Ђв”Ђ Voice Transcription в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/voice/transcribe', methods=['POST'])
@login_required
def transcribe_audio(user):
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    openai_key = os.environ.get('OPENAI_KEY', '')
    if not openai_key:
        return jsonify({'error': 'Voice transcription not configured'}), 503

    audio_file = request.files['audio']
    try:
        resp = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {openai_key}'},
            files={'file': (audio_file.filename or 'audio.webm', audio_file.stream, audio_file.content_type)},
            data={'model': 'whisper-1'},
            timeout=30
        )
        if resp.ok:
            return jsonify({'text': resp.json().get('text', '')})
        return jsonify({'error': f'Transcription failed: {resp.status_code}'}), resp.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# в”Ђв”Ђ Website Preview в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/preview/url', methods=['POST'])
@login_required
def preview_url(user):
    d   = request.get_json(silent=True) or {}
    url = d.get('url', '').strip()
    if not url:
        return jsonify({'error': 'URL required'}), 400
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    screenshot_key = os.environ.get('SCREENSHOT_API_KEY', '')
    if screenshot_key:
        try:
            resp = requests.get(
                'https://shot.screenshotapi.net/screenshot',
                params={'token': screenshot_key, 'url': url, 'width': 1280,
                        'height': 720, 'output': 'json', 'file_type': 'png'},
                timeout=30
            )
            if resp.ok:
                data = resp.json()
                return jsonify({'success': True, 'screenshot': data.get('screenshot', ''), 'url': url})
        except Exception:
            pass

    # Fallback placeholder
    return jsonify({
        'success': True,
        'screenshot': f'https://via.placeholder.com/1280x720/f7f7f8/888888?text={url[:50]}',
        'url': url,
        'fallback': True,
    })

# в”Ђв”Ђ Health check в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
@app.route('/api/health')
def health():
    groq_key = os.environ.get('OPENROUTER_KEY', '')
    return jsonify({
        'status': 'ok',
        'groq': bool(groq_key),
        'db': 'sqlite' if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI'] else 'postgres',
        'timestamp': datetime.utcnow().isoformat(),
    })

# в”Ђв”Ђ Run в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# ── Operator: Ban / Unban / Credits / Get User ────────────────────────
@app.route('/api/operator/user/<int:uid>/ban', methods=['POST'])
@operator_required
def operator_ban_user(user, uid):
    target = db.session.get(User, uid)
    if not target: return jsonify({'error': 'User not found'}), 404
    target.is_banned = True
    db.session.commit()
    return jsonify({'ok': True, 'user': target.to_dict()})

@app.route('/api/operator/user/<int:uid>/unban', methods=['POST'])
@operator_required
def operator_unban_user(user, uid):
    target = db.session.get(User, uid)
    if not target: return jsonify({'error': 'User not found'}), 404
    target.is_banned = False
    db.session.commit()
    return jsonify({'ok': True, 'user': target.to_dict()})

@app.route('/api/operator/user/<int:uid>/credits', methods=['POST'])
@operator_required
def operator_set_credits(user, uid):
    target = db.session.get(User, uid)
    if not target: return jsonify({'error': 'User not found'}), 404
    d = request.get_json(silent=True) or {}
    amount = int(d.get('amount', 0))
    mode = d.get('mode', 'add')  # add | set
    if mode == 'set':
        target.credits = max(0, amount)
    else:
        target.credits = max(0, (target.credits or 0) + amount)
    db.session.commit()
    return jsonify({'ok': True, 'user': target.to_dict()})

@app.route('/api/operator/user/<int:uid>', methods=['GET'])
@operator_required
def operator_get_user(user, uid):
    target = db.session.get(User, uid)
    if not target: return jsonify({'error': 'Not found'}), 404
    return jsonify(target.to_dict())

# ── Pay: Checkout / Webhook ───────────────────────────────────────────
@app.route('/api/pay/checkout', methods=['POST'])
@login_required
def pay_checkout(user):
    d = request.get_json(silent=True) or {}
    plan = d.get('plan', 'pro')
    credits_map = {'starter': 100, 'pro': 500, 'max': 2000, 'ultra': -1}
    price_map = {'starter': 99, 'pro': 299, 'max': 999, 'ultra': 1999}  # in rubles
    if plan not in credits_map:
        return jsonify({'error': 'Invalid plan'}), 400
    pay_url = os.environ.get('AICHATPAY_URL', 'https://pay.aichat.ru')
    return jsonify({
        'url': f'{pay_url}/checkout?plan={plan}&user_id={user.id}&amount={price_map[plan]}',
        'plan': plan,
        'credits': credits_map[plan],
        'price': price_map[plan]
    })

@app.route('/api/pay/webhook', methods=['POST'])
def pay_webhook():
    d = request.get_json(silent=True) or {}
    secret = os.environ.get('AICHATPAY_SECRET', '')
    if secret and d.get('secret') != secret:
        return jsonify({'error': 'Invalid secret'}), 403
    user_id = d.get('user_id')
    plan = d.get('plan', 'pro')
    credits_map = {'starter': 100, 'pro': 500, 'max': 2000, 'ultra': -1}
    if not user_id: return jsonify({'error': 'No user_id'}), 400
    user = db.session.get(User, int(user_id))
    if not user: return jsonify({'error': 'User not found'}), 404
    if plan == 'ultra':
        user.plan = 'ultra'
        user.credits = -1
    else:
        user.credits = (user.credits or 0) + credits_map.get(plan, 0)
        user.plan = plan
    db.session.commit()
    return jsonify({'ok': True})

# ── Reset all users (operator only) ──────────────────────────────────
@app.route('/api/operator/reset-all-users', methods=['POST'])
@operator_required
def reset_all_users(user):
    d = request.get_json(silent=True) or {}
    if d.get('confirm') != 'RESET_ALL':
        return jsonify({'error': 'Send confirm: RESET_ALL'}), 400
    try:
        count = User.query.count()
        # Delete all sessions and messages first
        ChatMessage.query.delete()
        ChatSession.query.delete()
        User.query.delete()
        db.session.commit()
        return jsonify({'ok': True, 'deleted': count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

