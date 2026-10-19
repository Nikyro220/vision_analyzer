from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from flask_babel import Babel

db = SQLAlchemy()
babel = Babel()
login_manager = LoginManager()
csrf = CSRFProtect()
migrate = Migrate()
