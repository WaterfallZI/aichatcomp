"""
AI Chat Company - Flask Server with Cursor API
Features: Auth, Cursor AI proxy, Sessions, Credits, Operator
"""
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, secrets, requests, traceback, json, time

app = Flask(__name__, static_folder='.')

# -- Config --
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

# -- Cursor API Config --
# Cursor API uses cookie-based auth to access claude/gpt-4o/etc models
CURSOR_API_URL = 'https://api2.cursor.sh/aiserver.v1.AiService/StreamChat'
CURSOR_CHAT_URL = 'https://api2.cursor.sh/v1/chat/completions'
NOMCHAT_URL = os.environ.get('NOMCHAT_URL', 'https://nomchat-id.up.railway.app')
OPERATOR_EMAIL = os.environ.get('OPERATOR_EMAIL', 'ai@com.ru')

# Available Cursor models
CURSOR_MODELS = {
    'claude-3-7-sonnet': 'claude-3-7-sonnet-20250219',
    'claude-3-5-sonnet': 'claude-3-5-sonnet-20241022',
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4-turbo': 'gpt-4-turbo',
    'o1-mini': 'o1-mini',
    'o3-mini': 'o3-mini',
    'gemini-2.0-flash': 'gemini-2.0-flash-exp',
    'deepseek-v3': 'deepseek-chat',
    'deepseek-r1': 'deepseek-reasoner',
}

PLANS = {
    'free':  {'credits': 50,   'bonus': 300,  'label': 'Free'},
    'pro':   {'credits': 1000, 'bonus': 0,    'label': 'Pro'},
    'max':   {'credits': 5000, 'bonus': 0,    'label': 'Max'},
    'ultra': {'credits': -1,   'bonus': 0,    'label': 'Ultra'},
}

# -- Models --
class User(db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    email          = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username       = db.Column(db.String(80), nullable=False)
    password_hash  = db.Column(db.String(256))
    nomchat_id     = db.Column(db.String(64), index=True)
    nomchat_username = db.Column(db.String(80))
    nomchat_avatar = db.Column(db.String(10), default='🦊')
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
            'nomchat_avatar': self.nomchat_avatar or '🦊',
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
    role       = db.Column(db.String(20), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            'id': self.id, 'role': self.role, 'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

# -- DB Init --
with app.app_context():
    db.create_all()

# -- Rate limiting --
_rate_store: dict = {}

def rate_limit(key: str, max_calls: int = 10, window: int = 60) -> bool:
    now = time.time()
    calls = [t for t in _rate_store.get(key, []) if now - t < window]
    if len(calls) >= max_calls:
        return False
    calls.append(now)
    _rate_store[key] = calls
    return True

# -- Auth decorators --
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

# -- Static files --
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    try:
        return send_from_directory('.', filename)
    except Exception:
        return jsonify({'error': 'Not found'}), 404

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f'500: {e}\n{traceback.format_exc()}')
    return jsonify({'error': 'Internal server error'}), 500

# -- Auth: Register --
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
    if len(username) < 2 or len(username) > 50:
        return jsonify({'error': 'Username must be 2-50 characters'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    try:
        user = User(email=email, username=username, credits=50, bonus_credits=300)
        user.set_password(password)
        if email == OPERATOR_EMAIL:
            user.is_operator = True
        db.session.add(user)
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict(), 'is_new': True})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Register error: {e}')
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
        if user.is_banned:
            return jsonify({'error': 'Account banned'}), 403

        user.last_login = datetime.utcnow()
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f'Login error: {e}')
        return jsonify({'error': 'Login failed'}), 500


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
                nomchat_avatar=nc_user.get('avatar', '🦊'),
            )
            db.session.add(user)
        else:
            user.nomchat_id       = str(nc_user.get('id', ''))
            user.nomchat_username = nc_user.get('username', user.nomchat_username)
            user.nomchat_avatar   = nc_user.get('avatar', user.nomchat_avatar or '🦊')

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

# -- Cursor API proxy --
def get_cursor_cookie():
    """Get Cursor cookie from env. Supports comma-separated multiple cookies for rotation."""
    cookies = os.environ.get('CURSOR_COOKIE', '')
    if not cookies:
        return None
    cookie_list = [c.strip() for c in cookies.split(',') if c.strip()]
    if not cookie_list:
        return None
    # Rotate cookies to distribute load
    idx = int(time.time()) % len(cookie_list)
    return cookie_list[idx]


def call_cursor_api(messages, model='claude-3-7-sonnet', stream=False):
    """Call Cursor API with OpenAI-compatible interface."""
    cookie = get_cursor_cookie()
    if not cookie:
        raise ValueError('CURSOR_COOKIE not configured')

    # Map model name to Cursor model
    cursor_model = CURSOR_MODELS.get(model, model)

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {cookie}',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }

    payload = {
        'model': cursor_model,
        'messages': messages,
        'stream': stream,
        'temperature': 0.7,
        'max_tokens': 4096,
    }

    resp = requests.post(
        CURSOR_CHAT_URL,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=60
    )
    return resp


@app.route('/api/config/models')
def get_models():
    """Return available models."""
    return jsonify({
        'models': list(CURSOR_MODELS.keys()),
        'default': 'claude-3-7-sonnet'
    })


@app.route('/api/chat', methods=['POST'])
@login_required
def chat(user):
    if not user.has_credits():
        return jsonify({'error': 'no_credits', 'message': 'No credits remaining.'}), 402

    cookie = get_cursor_cookie()
    if not cookie:
        return jsonify({'error': 'AI service not configured. Set CURSOR_COOKIE env variable.'}), 503

    d = request.get_json(silent=True) or {}
    messages = d.get('messages', [])
    model    = d.get('model', 'claude-3-7-sonnet')
    stream   = d.get('stream', False)

    # Validate model
    if model not in CURSOR_MODELS:
        model = 'claude-3-7-sonnet'

    # Filter messages
    clean_msgs = []
    for m in messages:
        role    = m.get('role', 'user')
        content = m.get('content', '')
        if role not in ('system', 'user', 'assistant'):
            continue
        if isinstance(content, str) and content.strip():
            clean_msgs.append({'role': role, 'content': content.strip()})
        elif isinstance(content, list):
            clean_msgs.append({'role': role, 'content': content})

    if not clean_msgs:
        return jsonify({'error': 'No messages provided'}), 400

    try:
        if stream:
            def generate():
                try:
                    resp = call_cursor_api(clean_msgs, model, stream=True)
                    if not resp.ok:
                        err_text = resp.text[:200]
                        yield f"data: {json.dumps({'error': f'Cursor API error {resp.status_code}: {err_text}'})}\n\n"
                        return
                    for line in resp.iter_lines():
                        if line:
                            decoded = line.decode('utf-8') if isinstance(line, bytes) else line
                            yield decoded + '\n\n'
                    user.deduct()
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(
                stream_with_context(generate()),
                content_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'}
            )
        else:
            resp = call_cursor_api(clean_msgs, model, stream=False)
            if not resp.ok:
                err_msg = f'Cursor API error {resp.status_code}'
                try:
                    err_data = resp.json()
                    err_msg = err_data.get('error', {}).get('message', err_msg) if isinstance(err_data.get('error'), dict) else err_data.get('error', err_msg)
                except Exception:
                    pass
                app.logger.warning(f'Cursor error: {resp.status_code} {err_msg}')
                return jsonify({'error': err_msg}), resp.status_code

            result = resp.json()
            user.deduct()
            total = -1 if user.plan == 'ultra' else (user.credits or 0) + (user.bonus_credits or 0)
            result['credits_remaining'] = total
            return jsonify(result)

    except requests.Timeout:
        return jsonify({'error': 'AI request timed out. Please try again.'}), 504
    except Exception as e:
        app.logger.error(f'Chat error: {e}')
        return jsonify({'error': str(e)}), 500

# -- User: Update / Delete --
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


# -- Plans --
@app.route('/api/plans')
def get_plans():
    return jsonify(PLANS)


# -- Chat Sessions (operator) --
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

# -- Operator --
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
    mode = d.get('mode', 'add')
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

# -- Pay --
@app.route('/api/pay/checkout', methods=['POST'])
@login_required
def pay_checkout(user):
    d = request.get_json(silent=True) or {}
    plan = d.get('plan', 'pro')
    credits_map = {'starter': 100, 'pro': 500, 'max': 2000, 'ultra': -1}
    price_map = {'starter': 99, 'pro': 299, 'max': 999, 'ultra': 1999}
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


# -- Health check --
@app.route('/api/health')
def health():
    cookie = get_cursor_cookie()
    return jsonify({
        'status': 'ok',
        'cursor_api': bool(cookie),
        'cursor_models': list(CURSOR_MODELS.keys()),
        'db': 'sqlite' if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI'] else 'postgres',
        'timestamp': datetime.utcnow().isoformat(),
    })


# -- Run --
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
