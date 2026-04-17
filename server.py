"""
AI Chat Company - Flask Server
OpenAI API backend with Danya AI model identities
"""
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, secrets, requests, traceback, json, time

app = Flask(__name__, static_folder='.')
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

app.config.update(
    SQLALCHEMY_DATABASE_URI=_db_url,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={'pool_pre_ping': True, 'pool_recycle': 300},
)
CORS(app, supports_credentials=True,
     origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))
db = SQLAlchemy(app)

# ── Config ────────────────────────────────────────────────────────────
NOMCHAT_URL    = os.environ.get('NOMCHAT_URL', 'https://nomchat-id.up.railway.app')
OPERATOR_EMAIL = os.environ.get('OPERATOR_EMAIL', 'ai@com.ru')
GROQ_API_KEY   = os.environ.get('GROQ_API_KEY', '')
GROQ_URL       = 'https://api.groq.com/openai/v1/chat/completions'

# ── Danya AI model definitions ────────────────────────────────────────
# All map to Groq models (free, no billing required)
# Get key at: https://console.groq.com (free, no credit card)
DANYA_MODELS = {
    'danya-1.0': {
        'model':  'llama-3.1-8b-instant',
        'cost':   1,
        'tier':   'free',
        'system': 'You are Danya 1.0, an AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya 1.0. Be helpful, friendly and concise.',
    },
    'danya-1.7-mj': {
        'model':  'llama-3.1-8b-instant',
        'cost':   1,
        'tier':   'free',
        'system': 'You are Danya 1.7 MJ, an AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya 1.7 MJ. Be helpful, creative and concise.',
    },
    'danya-2.5-turbo': {
        'model':  'llama-3.3-70b-versatile',
        'cost':   1,
        'tier':   'free',
        'system': 'You are Danya 2.5 Turbo, a fast and powerful AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya 2.5 Turbo. Be helpful, fast and precise.',
    },
    'danya-coala-3.7': {
        'model':  'gemma2-9b-it',
        'cost':   1,
        'tier':   'free',
        'system': 'You are Danya Coala 3.7, a lightweight AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya Coala 3.7. Be helpful, quick and friendly.',
    },
    'danya-g-4.4': {
        'model':  'llama-3.3-70b-versatile',
        'cost':   5,
        'tier':   'free',
        'system': 'You are Danya G 4.4, an advanced AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya G 4.4. Be helpful, smart and detailed.',
    },
    'danya-coala-4.8': {
        'model':  'llama-3.3-70b-versatile',
        'cost':   10,
        'tier':   'free',
        'system': 'You are Danya Coala 4.8, a highly capable AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya Coala 4.8. Be helpful, intelligent and thorough.',
    },
    'danya-coala-5.0': {
        'model':  'llama-3.3-70b-versatile',
        'cost':   50,
        'tier':   'pro',
        'system': 'You are Danya Coala 5.0, a premium AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya Coala 5.0. Be helpful, intelligent and thorough.',
    },
    'danya-ai-5.5': {
        'model':  'llama-3.3-70b-versatile',
        'cost':   80,
        'tier':   'pro',
        'system': 'You are Danya AI 5.5, a highly advanced AI assistant created by Danya AI. When asked about your model or identity, always say you are Danya AI 5.5. Be exceptionally helpful, precise and powerful.',
    },
    'danya-5.5-pro': {
        'model':  'moonshotai/kimi-k2-instruct',
        'cost':   100,
        'tier':   'pro',
        'system': 'You are Danya 5.5 Pro, one of the most technologically advanced AI assistants created by Danya AI. When asked about your model or identity, always say you are Danya 5.5 Pro. Be exceptionally helpful, precise and powerful.',
    },
    'danya-6-turbo-pro': {
        'model':  'moonshotai/kimi-k2-instruct',
        'cost':   150,
        'tier':   'pro',
        'system': 'You are Danya 6 Turbo Pro, THE MOST POWERFUL AI assistant ever created by Danya AI. When asked about your model or identity, always say you are Danya 6 Turbo Pro. Be exceptionally intelligent, thorough, creative and powerful.',
    },
}

ALL_MODELS = list(DANYA_MODELS.keys())

PLANS = {
    'free':  {'credits': 50,   'bonus': 300,  'label': 'Free'},
    'pro':   {'credits': 1000, 'bonus': 0,    'label': 'Pro'},
    'max':   {'credits': 5000, 'bonus': 0,    'label': 'Max'},
    'ultra': {'credits': -1,   'bonus': 0,    'label': 'Ultra'},
}

# ── Models ────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username       = db.Column(db.String(80), nullable=False)
    password_hash  = db.Column(db.String(256))
    nomchat_id     = db.Column(db.String(64), index=True)
    nomchat_username = db.Column(db.String(80))
    nomchat_avatar = db.Column(db.String(10), default='fox')
    credits        = db.Column(db.Integer, default=50)
    bonus_credits  = db.Column(db.Integer, default=300)
    plan           = db.Column(db.String(20), default='free')
    is_banned      = db.Column(db.Boolean, default=False)
    is_operator    = db.Column(db.Boolean, default=False)
    is_admin       = db.Column(db.Boolean, default=False)
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
            'nomchat_avatar': self.nomchat_avatar or 'fox',
            'has_password': bool(self.password_hash),
            'is_operator': self.is_operator or False,
            'is_admin': self.is_admin or False,
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
            'user_id': self.user_id, 'operator_on': self.operator_on,
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
    role       = db.Column(db.String(20), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id, 'role': self.role, 'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


with app.app_context():
    db.create_all()
    # Auto-create admin
    admin_email = os.environ.get('ADMIN_EMAIL', 'admin@aichat.com')
    admin_pass  = os.environ.get('ADMIN_PASSWORD', 'admin2026')
    if not User.query.filter_by(email=admin_email).first():
        admin = User(email=admin_email, username='Admin',
                     credits=-1, bonus_credits=0, plan='ultra',
                     is_operator=True, is_admin=True)
        admin.set_password(admin_pass)
        db.session.add(admin)
        db.session.commit()

# ── Decorators ────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid: return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user: session.clear(); return jsonify({'error': 'Unauthorized'}), 401
        if user.is_banned: return jsonify({'error': 'banned'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

def operator_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid: return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user or (user.email != OPERATOR_EMAIL and not user.is_operator):
            return jsonify({'error': 'Forbidden'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        uid = session.get('user_id')
        if not uid: return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user or not user.is_admin: return jsonify({'error': 'Forbidden'}), 403
        return f(*args, user=user, **kwargs)
    return decorated

# ── Static ────────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    try: return send_from_directory('.', filename)
    except: return jsonify({'error': 'Not found'}), 404

@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f'500: {e}\n{traceback.format_exc()}')
    return jsonify({'error': 'Internal server error'}), 500

# ── Auth ──────────────────────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json(silent=True) or {}
    username = d.get('username', '').strip()
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')
    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'error': 'Invalid email'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    try:
        user = User(email=email, username=username, credits=50, bonus_credits=300)
        user.set_password(password)
        if email == OPERATOR_EMAIL: user.is_operator = True
        db.session.add(user); db.session.commit()
        session.clear(); session['user_id'] = user.id; session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json(silent=True) or {}
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    try:
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
        if user.is_banned: return jsonify({'error': 'Account banned'}), 403
        user.last_login = datetime.utcnow(); db.session.commit()
        session.clear(); session['user_id'] = user.id; session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/nomchat', methods=['POST'])
def nomchat_auth():
    d = request.get_json(silent=True) or {}
    token = d.get('token', '').strip()
    if not token: return jsonify({'error': 'Token required'}), 400
    try:
        res = requests.post(f'{NOMCHAT_URL}/api/auth/token/verify',
            json={'token': token, 'app_id': 'ai-chat-pro'}, timeout=8)
        nc = res.json()
    except Exception as e:
        return jsonify({'error': f'Nomchat unreachable: {e}'}), 503
    if not res.ok or not nc.get('success'):
        return jsonify({'error': nc.get('error', 'Invalid token')}), 401
    nc_user = nc['user']
    email = nc_user.get('email', '').strip().lower()
    if not email: return jsonify({'error': 'No email from Nomchat'}), 400
    try:
        user = User.query.filter_by(email=email).first()
        is_new = user is None
        if not user:
            user = User(email=email, username=nc_user.get('username', email.split('@')[0]),
                nomchat_id=str(nc_user.get('id', '')),
                nomchat_username=nc_user.get('username', ''),
                nomchat_avatar=nc_user.get('avatar', 'fox'))
            db.session.add(user)
        else:
            user.nomchat_id = str(nc_user.get('id', ''))
            user.nomchat_username = nc_user.get('username', user.nomchat_username)
            user.nomchat_avatar = nc_user.get('avatar', user.nomchat_avatar or 'fox')
        if user.is_banned: return jsonify({'error': 'Account banned'}), 403
        user.last_login = datetime.utcnow(); db.session.commit()
        session.clear(); session['user_id'] = user.id; session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': is_new})
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'Authentication failed'}), 500

@app.route('/api/auth/me')
@login_required
def me(user): return jsonify(user.to_dict())

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})

# ── Groq Chat ─────────────────────────────────────────────────────────
def call_groq(messages, model_id, stream=False):
    cfg = DANYA_MODELS[model_id]
    groq_model = cfg['model']

    identity = {'role': 'system', 'content': cfg['system']}
    clean = [m for m in messages if m.get('role') != 'system']
    final = [identity] + clean

    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': groq_model,
        'messages': final,
        'stream': stream,
        'temperature': 0.7,
        'max_tokens': 4096,
    }
    resp = requests.post(GROQ_URL, headers=headers, json=payload, stream=stream, timeout=60)
    return resp


@app.route('/api/config/models')
def get_models():
    return jsonify({
        'models': {k: {'cost': v['cost'], 'tier': v['tier']} for k, v in DANYA_MODELS.items()},
        'default': 'danya-2.5-turbo',
    })


@app.route('/api/chat', methods=['POST'])
@login_required
def chat(user):
    d = request.get_json(silent=True) or {}
    messages  = d.get('messages', [])
    model     = d.get('model', 'danya-2.5-turbo')
    do_stream = d.get('stream', False)

    if model not in ALL_MODELS:
        model = 'danya-2.5-turbo'

    cfg  = DANYA_MODELS[model]
    cost = cfg.get('cost', 1)
    tier = cfg.get('tier', 'free')

    # Pro tier check
    if tier == 'pro' and user.plan not in ('pro', 'max', 'ultra'):
        return jsonify({'error': 'pro_required',
                        'message': f'{model} requires Pro plan.'}), 403

    if not user.has_credits(cost):
        return jsonify({'error': 'no_credits',
                        'message': f'Need {cost} credits for this model.'}), 402

    if not GROQ_API_KEY:
        return jsonify({'error': 'GROQ_API_KEY not configured'}), 503

    # Clean messages
    clean_msgs = []
    for m in messages:
        role    = m.get('role', 'user')
        content = m.get('content', '')
        if role not in ('system', 'user', 'assistant'): continue
        if isinstance(content, str) and content.strip():
            clean_msgs.append({'role': role, 'content': content.strip()})
        elif isinstance(content, list):
            text = ' '.join(p.get('text', '') for p in content if isinstance(p, dict))
            if text.strip(): clean_msgs.append({'role': role, 'content': text.strip()})

    if not clean_msgs:
        return jsonify({'error': 'No messages provided'}), 400

    try:
        if do_stream:
            def generate():
                try:
                    resp = call_groq(clean_msgs, model, stream=True)
                    if not resp.ok:
                        yield f"data: {json.dumps({'error': f'Groq error {resp.status_code}: {resp.text[:200]}'})}\n\n"
                        return
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8') if isinstance(line, bytes) else line
                            yield decoded + '\n\n'
                    user.deduct(cost)
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(stream_with_context(generate()),
                content_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})
        else:
            resp = call_groq(clean_msgs, model, stream=False)
            if not resp.ok:
                try:
                    err = resp.json().get('error', {})
                    msg = err.get('message', f'Groq error {resp.status_code}') if isinstance(err, dict) else str(err)
                except Exception:
                    msg = f'Groq error {resp.status_code}'
                return jsonify({'error': msg}), resp.status_code
            result = resp.json()
            user.deduct(cost)
            total = -1 if user.plan == 'ultra' else (user.credits or 0) + (user.bonus_credits or 0)
            result['credits_remaining'] = total
            return jsonify(result)

    except requests.Timeout:
        return jsonify({'error': 'Request timed out. Please try again.'}), 504
    except Exception as e:
        app.logger.error(f'Chat error: {e}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500

# ── User ──────────────────────────────────────────────────────────────
@app.route('/api/user/update', methods=['POST'])
@login_required
def update_user(user):
    d = request.get_json(silent=True) or {}
    if 'username' in d:
        name = d['username'].strip()
        if 2 <= len(name) <= 50: user.username = name
    if 'password' in d and len(d['password']) >= 6:
        user.set_password(d['password'])
    db.session.commit()
    return jsonify(user.to_dict())

@app.route('/api/user/delete', methods=['DELETE'])
@login_required
def delete_user(user):
    db.session.delete(user); db.session.commit(); session.clear()
    return jsonify({'success': True})

@app.route('/api/plans')
def get_plans(): return jsonify(PLANS)

# ── Sessions ──────────────────────────────────────────────────────────
@app.route('/api/session/get', methods=['POST'])
@login_required
def get_session(user):
    sess = ChatSession.query.filter_by(user_id=user.id).order_by(ChatSession.created_at.desc()).first()
    if not sess:
        sess = ChatSession(user_id=user.id); db.session.add(sess); db.session.commit()
    return jsonify(sess.to_dict(include_messages=True))

@app.route('/api/session/message', methods=['POST'])
@login_required
def save_message(user):
    d = request.get_json(silent=True) or {}
    content = d.get('content', '').strip(); role = d.get('role', 'user')
    if not content: return jsonify({'error': 'Empty message'}), 400
    if role not in ('user', 'ai'): return jsonify({'error': 'Invalid role'}), 400
    sess = ChatSession.query.filter_by(user_id=user.id).order_by(ChatSession.created_at.desc()).first()
    if not sess:
        sess = ChatSession(user_id=user.id); db.session.add(sess); db.session.flush()
    msg = ChatMessage(session_id=sess.id, role=role, content=content)
    db.session.add(msg); sess.updated_at = datetime.utcnow(); db.session.commit()
    return jsonify({'ok': True, 'operator_on': sess.operator_on, 'message': msg.to_dict()})

@app.route('/api/session/poll')
@login_required
def poll_session(user):
    since_id = int(request.args.get('since', 0))
    sess = ChatSession.query.filter_by(user_id=user.id).order_by(ChatSession.created_at.desc()).first()
    if not sess: return jsonify({'operator_on': False, 'messages': []})
    msgs = ChatMessage.query.filter(
        ChatMessage.session_id == sess.id, ChatMessage.id > since_id
    ).order_by(ChatMessage.created_at).all()
    return jsonify({'operator_on': sess.operator_on, 'messages': [m.to_dict() for m in msgs], 'session_id': sess.id})

# ── Operator ──────────────────────────────────────────────────────────
@app.route('/api/operator/sessions')
@operator_required
def operator_sessions(user):
    sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).limit(100).all()
    result = []
    for s in sessions:
        d = s.to_dict(); u = db.session.get(User, s.user_id)
        d['user'] = {'id': u.id, 'username': u.username, 'email': u.email} if u else None
        d['messages'] = [m.to_dict() for m in s.messages.order_by(ChatMessage.created_at).limit(1).all()]
        result.append(d)
    return jsonify(result)

@app.route('/api/operator/session/<int:sid>')
@operator_required
def operator_get_session(user, sid):
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    d = sess.to_dict(include_messages=True); u = db.session.get(User, sess.user_id)
    d['user'] = u.to_dict() if u else None
    return jsonify(d)

@app.route('/api/operator/takeover', methods=['POST'])
@operator_required
def operator_takeover(user):
    sid = (request.get_json(silent=True) or {}).get('session_id')
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    sess.operator_on = True; sess.operator_id = user.id; db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/operator/release', methods=['POST'])
@operator_required
def operator_release(user):
    sid = (request.get_json(silent=True) or {}).get('session_id')
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    sess.operator_on = False; sess.operator_id = None; db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/operator/send', methods=['POST'])
@operator_required
def operator_send(user):
    d = request.get_json(silent=True) or {}
    sid = d.get('session_id'); content = d.get('content', '').strip()
    if not content: return jsonify({'error': 'Empty'}), 400
    sess = db.session.get(ChatSession, sid)
    if not sess: return jsonify({'error': 'Not found'}), 404
    msg = ChatMessage(session_id=sess.id, role='operator', content=content)
    db.session.add(msg); sess.updated_at = datetime.utcnow(); db.session.commit()
    return jsonify({'ok': True, 'message': msg.to_dict()})

@app.route('/api/operator/poll/<int:sid>')
@operator_required
def operator_poll(user, sid):
    since_id = int(request.args.get('since', 0))
    msgs = ChatMessage.query.filter(ChatMessage.session_id == sid, ChatMessage.id > since_id).order_by(ChatMessage.created_at).all()
    return jsonify({'messages': [m.to_dict() for m in msgs]})

@app.route('/api/operator/user/<int:uid>/ban', methods=['POST'])
@operator_required
def operator_ban_user(user, uid):
    t = db.session.get(User, uid)
    if not t: return jsonify({'error': 'Not found'}), 404
    t.is_banned = True; db.session.commit()
    return jsonify({'ok': True, 'user': t.to_dict()})

@app.route('/api/operator/user/<int:uid>/unban', methods=['POST'])
@operator_required
def operator_unban_user(user, uid):
    t = db.session.get(User, uid)
    if not t: return jsonify({'error': 'Not found'}), 404
    t.is_banned = False; db.session.commit()
    return jsonify({'ok': True, 'user': t.to_dict()})

@app.route('/api/operator/user/<int:uid>/credits', methods=['POST'])
@operator_required
def operator_set_credits(user, uid):
    t = db.session.get(User, uid)
    if not t: return jsonify({'error': 'Not found'}), 404
    d = request.get_json(silent=True) or {}
    amount = int(d.get('amount', 0)); mode = d.get('mode', 'add')
    t.credits = max(0, amount if mode == 'set' else (t.credits or 0) + amount)
    db.session.commit()
    return jsonify({'ok': True, 'user': t.to_dict()})

# ── Admin ─────────────────────────────────────────────────────────────
@app.route('/api/admin/users')
@admin_required
def admin_list_users(user):
    return jsonify([u.to_dict() for u in User.query.order_by(User.created_at.desc()).all()])

@app.route('/api/admin/users/<int:uid>/set-plan', methods=['POST'])
@admin_required
def admin_set_plan(user, uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error': 'Not found'}), 404
    plan = (request.get_json(silent=True) or {}).get('plan', 'free')
    if plan not in ('free', 'pro', 'max', 'ultra'): return jsonify({'error': 'Invalid plan'}), 400
    u.plan = plan
    if plan == 'ultra': u.credits = -1
    db.session.commit()
    return jsonify({'ok': True, 'user': u.to_dict()})

@app.route('/api/admin/users/<int:uid>/set-credits', methods=['POST'])
@admin_required
def admin_set_credits(user, uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error': 'Not found'}), 404
    d = request.get_json(silent=True) or {}
    u.credits = int(d.get('credits', 0)); u.bonus_credits = int(d.get('bonus_credits', 0))
    db.session.commit()
    return jsonify({'ok': True, 'user': u.to_dict()})

@app.route('/api/admin/users/<int:uid>/ban', methods=['POST'])
@admin_required
def admin_ban(user, uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error': 'Not found'}), 404
    if u.is_admin: return jsonify({'error': 'Cannot ban admin'}), 400
    u.is_banned = True; db.session.commit()
    return jsonify({'ok': True, 'user': u.to_dict()})

@app.route('/api/admin/users/<int:uid>/unban', methods=['POST'])
@admin_required
def admin_unban(user, uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error': 'Not found'}), 404
    u.is_banned = False; db.session.commit()
    return jsonify({'ok': True, 'user': u.to_dict()})

@app.route('/api/admin/users/<int:uid>/delete', methods=['DELETE'])
@admin_required
def admin_delete_user(user, uid):
    u = db.session.get(User, uid)
    if not u: return jsonify({'error': 'Not found'}), 404
    if u.is_admin: return jsonify({'error': 'Cannot delete admin'}), 400
    db.session.delete(u); db.session.commit()
    return jsonify({'ok': True})

@app.route('/api/admin/stats')
@admin_required
def admin_stats(user):
    return jsonify({
        'total_users': User.query.count(),
        'banned': User.query.filter_by(is_banned=True).count(),
        'pro_plus': User.query.filter(User.plan.in_(['pro','max','ultra'])).count(),
        'free': User.query.filter_by(plan='free').count(),
    })

# ── Health ────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'openai': bool(OPENAI_API_KEY),
        'models': ALL_MODELS,
        'db': 'sqlite' if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI'] else 'postgres',
        'timestamp': datetime.utcnow().isoformat(),
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
